"""Tests for _purge_stale_hermes_modules — the class fix for stale
sys.modules breaking the gateway auto-restart after `hermes update`.

Field failure (2026-08-20, Teknium's Linux box): `hermes update` pulled a
checkout where hermes_cli/gateway.py newly imports `line_input` from
hermes_cli.cli_output, but the updater process had cli_output cached from
before that symbol existed. The function-level `from hermes_cli.gateway
import ...` in the restart phase raised ImportError, the whole phase
aborted, and the running gateway kept serving pre-update code.

The old mitigation (_UPDATE_RUNTIME_RELOAD_MODULES) reloaded 3 hardcoded
modules — re-fixed per symptom. The purge evicts EVERY cached module under
the Hermes package prefixes so later imports rebuild a self-consistent
module graph from the updated checkout.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from hermes_cli import main as cli_main
from hermes_cli import update_cmd


@pytest.fixture(autouse=True)
def _restore_sys_modules():
    """Snapshot & restore sys.modules around each test.

    The purge under test evicts real Hermes modules from the cache; later
    tests in the same process may hold references to the evicted module
    objects (e.g. `patch.object` targets), so put the originals back.
    """
    snapshot = dict(sys.modules)
    yield
    for name, mod in snapshot.items():
        sys.modules[name] = mod
    for name in list(sys.modules):
        if name not in snapshot:
            del sys.modules[name]


def _fake_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__stale_sentinel__ = True
    return mod


def test_purge_evicts_hermes_prefixed_modules():
    victims = [
        "hermes_cli.cli_output",
        "hermes_cli.gateway",
        "gateway.status",
        "tools.ansi_strip",
        "tui_gateway.server",
        "agent.memory_store",
    ]
    added = []
    for name in victims:
        if name not in sys.modules:
            sys.modules[name] = _fake_module(name)
            added.append(name)
    try:
        cli_main._purge_stale_hermes_modules()
        for name in victims:
            mod = sys.modules.get(name)
            assert mod is None or not getattr(mod, "__stale_sentinel__", False), (
                f"{name} survived the purge"
            )
    finally:
        for name in added:
            sys.modules.pop(name, None)


def test_purge_protects_executing_modules():
    # The updater's own modules must survive — they're running this code.
    cli_main._purge_stale_hermes_modules()
    assert sys.modules.get("hermes_cli.update_cmd") is update_cmd
    assert sys.modules.get("hermes_cli.main") is cli_main
    assert "hermes_cli" in sys.modules


def test_purge_preserves_in_flight_update_receipt(tmp_path):
    """A begun receipt must survive the purge and still be written.

    Field failure (2026-09-01): every SUCCESSFUL update lost its receipt.
    ``begin_update_receipt()`` runs at the top of the update; the purge
    after the pull evicted ``hermes_cli.update_receipt`` from sys.modules,
    so the updater's late ``from hermes_cli.update_receipt import
    finalize_update_receipt`` executed a FRESH module whose ``_current``
    singleton is None — the begun receipt was orphaned and never
    persisted. Failed updates exited before the purge, which is why only
    failures had receipts. The receipt module holds the update's in-flight
    state, so it belongs with the protected executing modules.

    Runs in a subprocess: the real purge evicts real modules, and an
    in-process run would leave the parent pytest process with a half-
    restored module graph (parent-package attributes diverge from
    sys.modules), leaking into unrelated tests.
    """
    import json
    import subprocess

    home = tmp_path / ".hermes"
    home.mkdir()
    code = (
        "import sys\n"
        "import hermes_cli.update_receipt as ur\n"
        "from hermes_cli import main as cli_main\n"
        "ur.begin_update_receipt()\n"
        "ur.record_step('pre_update_backup', False, 'disabled or failed')\n"
        "cli_main._purge_stale_hermes_modules()\n"
        "assert sys.modules.get('hermes_cli.update_receipt') is ur, (\n"
        "    'receipt module was evicted by the purge')\n"
        "assert ur._current is not None, 'receipt singleton orphaned by purge'\n"
        "from hermes_cli.update_receipt import finalize_update_receipt\n"
        "path = ur.finalize_update_receipt('success')\n"
        "assert path is not None and path.is_file(), (\n"
        "    'begun receipt was orphaned by the purge — finalize became a no-op')\n"
        "print(path)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "HERMES_HOME": str(home)},
    )
    assert result.returncode == 0, result.stderr
    receipt = Path(result.stdout.strip().splitlines()[-1])
    assert receipt.is_file()
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["outcome"] == "success"
    assert payload["steps"][0]["name"] == "pre_update_backup"


def test_purge_leaves_prefix_lookalikes_alone():
    # `gateway_foo` starts with the string prefix "gateway" but is NOT the
    # gateway package — the root-segment check must spare it.
    lookalikes = ["gatewayd", "toolshed", "agents_external"]
    added = []
    for name in lookalikes:
        if name not in sys.modules:
            sys.modules[name] = _fake_module(name)
            added.append(name)
    try:
        cli_main._purge_stale_hermes_modules()
        for name in lookalikes:
            assert name in sys.modules, f"{name} was wrongly purged"
    finally:
        for name in added:
            sys.modules.pop(name, None)


def test_purge_never_raises_on_weird_sys_modules():
    # Entries with None values (import machinery quirk) must not break it.
    sys.modules["hermes_cli._purge_test_none"] = None  # type: ignore[assignment]
    try:
        cli_main._purge_stale_hermes_modules()
    finally:
        sys.modules.pop("hermes_cli._purge_test_none", None)


def test_stale_symbol_scenario_end_to_end():
    """Reproduce the field failure shape: a cached module missing a symbol
    that freshly-imported code needs — purge, then re-import resolves it."""
    name = "hermes_cli.cli_output"
    real = sys.modules.get(name)
    # Install a stale stand-in WITHOUT line_input (pre-d0132b582 world).
    stale = types.ModuleType(name)
    sys.modules[name] = stale
    try:
        # The failure mode: importing the symbol from the stale cache dies.
        try:
            from hermes_cli.cli_output import line_input  # noqa: F401
            raised = False
        except ImportError:
            raised = True
        assert raised, "precondition: stale module must lack line_input"

        cli_main._purge_stale_hermes_modules()

        # Post-purge, the import resolves against real on-disk source.
        from hermes_cli.cli_output import line_input  # noqa: F401
    finally:
        sys.modules.pop(name, None)
        if real is not None:
            sys.modules[name] = real
