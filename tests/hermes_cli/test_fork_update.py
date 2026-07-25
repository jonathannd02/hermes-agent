"""Unit tests for hermes_cli.fork_update — custom-branch rebase flow.

The module's core promise is that the live checkout is never mutated: the
rebase happens inside a throwaway detached worktree. Assertions here check
*where* each git command ran and *what* it targeted, not just that some
command of the right name was issued — a check on the subcommand alone
still passes when the rebase is pointed at the live checkout or the wrong
upstream ref.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import fork_update as fu


def _cp(cmd, rc=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(cmd, rc, stdout=stdout, stderr=stderr)


def _git_side_effect(
    *,
    branch="custom-main",
    news=3,
    branch_sha="aaa111",
    new_sha="bbb222",
    remote_sha=None,
    rebase_rc=0,
    conflicted=None,
    worktree_add_rc=0,
    push_rc=0,
    push_stderr="",
    upstream_remote_ok=True,
    fetch_rc=0,
):
    """Simulate the git command sequences used by fork_update.

    ``branch_sha``/``new_sha`` default to distinct values so a refspec built
    from the wrong one is visible. Pass ``branch_sha=""`` to simulate an
    unresolvable branch, ``new_sha=""`` for a rebase that produces nothing,
    and ``remote_sha=""`` for a branch that does not exist on origin yet;
    ``remote_sha=None`` means origin already matches the local branch.
    """

    conflicted = conflicted or []
    remote_sha = remote_sha if remote_sha is not None else branch_sha

    def side_effect(git_cmd, cwd, *argv, check=False):
        args = list(argv)
        cmd = list(git_cmd) + args

        if args[:2] == ["rev-parse", "--abbrev-ref"]:
            return _cp(cmd, 0, f"{branch}\n")

        if args[:1] == ["remote"] and "get-url" in args:
            return _cp(cmd, 0 if upstream_remote_ok else 128)

        if args[:1] == ["fetch"]:
            return _cp(cmd, fetch_rc)

        if args[:1] == ["rev-list"] and "--count" in args:
            spec = args[-1]
            if spec.startswith("HEAD.."):
                return _cp(cmd, 0, f"{news}\n")
            if ".." in spec:
                # upstream/main..<new_sha> after rebase
                return _cp(cmd, 0, "5\n")
            return _cp(cmd, 0, "0\n")

        if args[:1] == ["rev-parse"] and "--verify" in args:
            ref = args[-1]
            if ref == "HEAD":
                return _cp(cmd, 0, f"{new_sha}\n")
            if ref == f"origin/{branch}":
                if remote_sha:
                    return _cp(cmd, 0, f"{remote_sha}\n")
                return _cp(cmd, 1)
            return _cp(cmd, 0, f"{branch_sha}\n")

        if args[:1] == ["config"]:
            return _cp(cmd, 0)

        if args[:2] == ["worktree", "remove"]:
            return _cp(cmd, 0)

        if args[:2] == ["worktree", "add"]:
            return _cp(cmd, worktree_add_rc, stderr="fatal: worktree add failed\n")

        if args[:2] == ["rebase", "--abort"]:
            return _cp(cmd, 0)

        if args[:1] == ["rebase"]:
            return _cp(cmd, rebase_rc, stderr="CONFLICT\n" if rebase_rc else "")

        if args[:1] == ["diff"] and "--diff-filter=U" in args:
            return _cp(cmd, 0, "\n".join(conflicted) + ("\n" if conflicted else ""))

        if args[:1] == ["push"]:
            return _cp(cmd, push_rc, stderr=push_stderr)

        if check:
            raise subprocess.CalledProcessError(1, cmd)
        return _cp(cmd, 0)

    return side_effect


def _calls(run):
    """Every git invocation as ``(cwd, argv)``."""
    return [(c.args[1], tuple(c.args[2:])) for c in run.call_args_list]


def _matching(run, *prefix):
    """Invocations whose argv starts with ``prefix``, as ``(cwd, argv)``."""
    return [item for item in _calls(run) if item[1][: len(prefix)] == prefix]


def _rebase_worktree(repo_root, branch="custom-main"):
    """The throwaway worktree path fork_update derives for ``branch``."""
    return repo_root.parent / f".hermes-fork-rebase-{branch}"


class TestMaybeRebaseForkBranchGuards:
    def test_explicit_branch_skips_fork_flow(self, tmp_path):
        args = SimpleNamespace(branch="main")
        with patch.object(fu, "_run") as run:
            assert fu.maybe_rebase_fork_branch(["git"], tmp_path, args) is None
            run.assert_not_called()

    def test_main_branch_skips_fork_flow(self, tmp_path):
        args = SimpleNamespace(branch=None)
        with patch.object(fu, "_run", side_effect=_git_side_effect(branch="main")):
            assert fu.maybe_rebase_fork_branch(["git"], tmp_path, args) is None

    def test_detached_head_skips_fork_flow(self, tmp_path):
        args = SimpleNamespace(branch=None)
        with patch.object(fu, "_run", side_effect=_git_side_effect(branch="HEAD")):
            assert fu.maybe_rebase_fork_branch(["git"], tmp_path, args) is None


class TestUpstreamNewsCount:
    def test_no_upstream_remote(self, tmp_path):
        with patch.object(
            fu, "_run", side_effect=_git_side_effect(upstream_remote_ok=False)
        ):
            assert fu.upstream_news_count(["git"], tmp_path) == -1

    def test_fetch_failure(self, tmp_path):
        with patch.object(fu, "_run", side_effect=_git_side_effect(fetch_rc=1)):
            assert fu.upstream_news_count(["git"], tmp_path) == -1

    def test_counts_missing_commits(self, tmp_path):
        with patch.object(fu, "_run", side_effect=_git_side_effect(news=7)) as run:
            assert fu.upstream_news_count(["git"], tmp_path) == 7
        assert _matching(run, "fetch", "upstream", "main", "--quiet")
        assert _matching(run, "rev-list", "--count", "HEAD..upstream/main")

    def test_unparsable_count_reports_unreachable(self, tmp_path):
        def side_effect(git_cmd, cwd, *argv, check=False):
            if argv[:1] == ("rev-list",):
                return _cp(list(argv), 0, "not-a-number\n")
            return _cp(list(argv), 0)

        with patch.object(fu, "_run", side_effect=side_effect):
            assert fu.upstream_news_count(["git"], tmp_path) == -1


class TestMaybeRebaseForkBranchPaths:
    def test_upstream_unreachable_returns_branch(self, tmp_path, capsys):
        args = SimpleNamespace(branch=None)
        with patch.object(
            fu, "_run", side_effect=_git_side_effect(upstream_remote_ok=False)
        ) as run:
            assert fu.maybe_rebase_fork_branch(["git"], tmp_path, args) == "custom-main"
        assert "Could not reach upstream" in capsys.readouterr().out
        # Nothing may be rewritten when upstream state is unknown.
        assert not _matching(run, "rebase")
        assert not _matching(run, "push")

    def test_already_up_to_date_syncs_origin_when_lagging(self, tmp_path, capsys):
        args = SimpleNamespace(branch=None)
        with patch.object(
            fu,
            "_run",
            side_effect=_git_side_effect(news=0, remote_sha="old999"),
        ) as run:
            assert fu.maybe_rebase_fork_branch(["git"], tmp_path, args) == "custom-main"
        assert "Synced origin/custom-main" in capsys.readouterr().out
        assert not _matching(run, "rebase"), "no rebase when upstream has nothing new"
        pushes = _matching(run, "push")
        assert pushes, "expected a push when origin lags"
        cwd, argv = pushes[0]
        assert cwd == tmp_path
        assert argv == (
            "push",
            "--force-with-lease",
            "origin",
            "aaa111:refs/heads/custom-main",
        )

    def test_already_up_to_date_skips_push_when_synced(self, tmp_path):
        args = SimpleNamespace(branch=None)
        with patch.object(
            fu,
            "_run",
            side_effect=_git_side_effect(news=0, branch_sha="same", remote_sha="same"),
        ) as run:
            assert fu.maybe_rebase_fork_branch(["git"], tmp_path, args) == "custom-main"
        assert not _matching(run, "push")

    def test_pushes_branch_missing_on_origin(self, tmp_path):
        args = SimpleNamespace(branch=None)
        with patch.object(
            fu, "_run", side_effect=_git_side_effect(news=0, remote_sha="")
        ) as run:
            assert fu.maybe_rebase_fork_branch(["git"], tmp_path, args) == "custom-main"
        assert _matching(run, "push"), "a branch absent from origin must be pushed"

    def test_rebase_runs_in_throwaway_worktree_not_live_checkout(self, tmp_path):
        args = SimpleNamespace(branch=None)
        with patch.object(fu, "_run", side_effect=_git_side_effect(news=3)) as run:
            fu.maybe_rebase_fork_branch(["git"], tmp_path, args)

        worktree = _rebase_worktree(tmp_path)
        rebases = _matching(run, "rebase")
        assert rebases == [(worktree, ("rebase", "upstream/main"))]

        # The live checkout may only be read from or have its worktree/refs
        # administered — never rewritten.
        forbidden = {"rebase", "reset", "checkout", "merge", "cherry-pick"}
        live = [argv for cwd, argv in _calls(run) if cwd == tmp_path]
        assert not [argv for argv in live if argv[0] in forbidden]

    def test_worktree_is_created_from_branch_tip_and_cleaned_up(self, tmp_path):
        args = SimpleNamespace(branch=None)
        with patch.object(fu, "_run", side_effect=_git_side_effect(news=3)) as run:
            fu.maybe_rebase_fork_branch(["git"], tmp_path, args)

        worktree = _rebase_worktree(tmp_path)
        assert _matching(run, "worktree", "add") == [
            (tmp_path, ("worktree", "add", "--detach", str(worktree), "aaa111"))
        ]

        sequence = [argv[:2] for _cwd, argv in _calls(run)]
        rebase_at = sequence.index(("rebase", "upstream/main"))
        removes = [i for i, argv in enumerate(sequence) if argv == ("worktree", "remove")]
        assert [i for i in removes if i < rebase_at], "stale worktree must be cleared first"
        assert [i for i in removes if i > rebase_at], "worktree must be removed afterwards"

    def test_successful_rebase_pushes_and_returns_branch(self, tmp_path, capsys):
        args = SimpleNamespace(branch=None)
        with patch.object(fu, "_run", side_effect=_git_side_effect(news=3)) as run:
            assert fu.maybe_rebase_fork_branch(["git"], tmp_path, args) == "custom-main"
        out = capsys.readouterr().out
        assert "rebasing 'custom-main' onto upstream/main" in out
        assert "Rebased 5 fork commit(s)" in out

        pushes = _matching(run, "push")
        assert len(pushes) == 1
        cwd, argv = pushes[0]
        assert cwd == tmp_path
        # The rebased HEAD is pushed, not the pre-rebase branch tip.
        assert argv == (
            "push",
            "--force-with-lease",
            "origin",
            "bbb222:refs/heads/custom-main",
        )

    def test_unresolvable_branch_exits(self, tmp_path, capsys):
        args = SimpleNamespace(branch=None)
        with patch.object(
            fu, "_run", side_effect=_git_side_effect(news=2, branch_sha="")
        ) as run:
            with pytest.raises(SystemExit) as exc:
                fu.maybe_rebase_fork_branch(["git"], tmp_path, args)
        assert exc.value.code == 1
        assert "Could not resolve branch 'custom-main'" in capsys.readouterr().out
        assert not _matching(run, "worktree", "add")

    def test_worktree_add_failure_exits(self, tmp_path, capsys):
        args = SimpleNamespace(branch=None)
        with patch.object(
            fu, "_run", side_effect=_git_side_effect(news=2, worktree_add_rc=1)
        ) as run:
            with pytest.raises(SystemExit) as exc:
                fu.maybe_rebase_fork_branch(["git"], tmp_path, args)
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "Could not create temporary rebase worktree" in out
        assert "fatal: worktree add failed" in out
        assert not _matching(run, "rebase")
        assert not _matching(run, "push")

    def test_empty_rebase_result_exits_without_pushing(self, tmp_path, capsys):
        args = SimpleNamespace(branch=None)
        with patch.object(
            fu, "_run", side_effect=_git_side_effect(news=2, new_sha="")
        ) as run:
            with pytest.raises(SystemExit) as exc:
                fu.maybe_rebase_fork_branch(["git"], tmp_path, args)
        assert exc.value.code == 1
        assert "Rebase produced no result" in capsys.readouterr().out
        assert not _matching(run, "push")

    def test_rebase_conflict_aborts_and_exits(self, tmp_path, capsys):
        args = SimpleNamespace(branch=None)
        with patch.object(
            fu,
            "_run",
            side_effect=_git_side_effect(
                news=2, rebase_rc=1, conflicted=["hermes_cli/main.py"]
            ),
        ) as run:
            with pytest.raises(SystemExit) as exc:
                fu.maybe_rebase_fork_branch(["git"], tmp_path, args)
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "hit conflicts" in out
        assert "hermes_cli/main.py" in out
        assert "git rebase upstream/main" in out

        worktree = _rebase_worktree(tmp_path)
        assert _matching(run, "rebase", "--abort") == [
            (worktree, ("rebase", "--abort"))
        ]
        assert not _matching(run, "push"), "a conflicted rebase must not reach origin"
        assert _matching(run, "worktree", "remove")

    def test_push_failure_exits(self, tmp_path, capsys):
        args = SimpleNamespace(branch=None)
        with patch.object(
            fu,
            "_run",
            side_effect=_git_side_effect(
                news=1, push_rc=1, push_stderr="error: failed to push some refs\n"
            ),
        ):
            with pytest.raises(SystemExit) as exc:
                fu.maybe_rebase_fork_branch(["git"], tmp_path, args)
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "Could not push rebased branch" in out
        assert "failed to push some refs" in out
        assert "bbb222:refs/heads/custom-main" in out, "manual command must be printed"


class TestEnsureRerere:
    def test_enables_rerere_config(self, tmp_path):
        with patch.object(fu, "_run", return_value=_cp(["git"], 0)) as run:
            fu._ensure_rerere(["git"], tmp_path)
        configs = [argv for _cwd, argv in _matching(run, "config")]
        assert ("config", "rerere.enabled", "true") in configs
        assert ("config", "rerere.autoUpdate", "true") in configs

    def test_enabled_before_any_rebase(self, tmp_path):
        """rerere must be on before the rebase, or resolutions are not replayed."""
        args = SimpleNamespace(branch=None)
        with patch.object(fu, "_run", side_effect=_git_side_effect(news=3)) as run:
            fu.maybe_rebase_fork_branch(["git"], tmp_path, args)
        sequence = [argv[:2] for _cwd, argv in _calls(run)]
        assert sequence.index(("config", "rerere.enabled")) < sequence.index(
            ("rebase", "upstream/main")
        )
