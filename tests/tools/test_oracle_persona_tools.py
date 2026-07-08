import json
import os
from pathlib import Path

import pytest

from tools import oracle_persona_tools as opt


def _seed_persona(tmp_path: Path) -> Path:
    root = tmp_path / "oracle-sources"
    (root / "personas" / "dejaru22").mkdir(parents=True, exist_ok=True)

    x_dir = root / "xitter" / "dejaru22" / "index"
    x_dir.mkdir(parents=True, exist_ok=True)
    x_file = x_dir / "posts.jsonl"
    x_rows = [
        {
            "source_id": "dejaru22:tweet:1",
            "persona_id": "dejaru22",
            "source_type": "x_post",
            "post_id": "1",
            "text": "Discipline beats mood every day.",
            "canonical_url": "https://x.com/DejaRu22/status/1",
        },
        {
            "source_id": "dejaru22:tweet:2",
            "persona_id": "dejaru22",
            "source_type": "x_post",
            "post_id": "2",
            "text": "Your environment programs your behavior.",
            "canonical_url": "https://x.com/DejaRu22/status/2",
        },
    ]
    x_file.write_text("".join(json.dumps(r) + "\n" for r in x_rows), encoding="utf-8")

    tg_dir = root / "telegram" / "rubisroundtable" / "index"
    tg_dir.mkdir(parents=True, exist_ok=True)
    tg_file = tg_dir / "messages.jsonl"
    tg_rows = [
        {
            "source_id": "telegram:rubisroundtable:msg:10",
            "persona_id": "dejaru22",
            "source_type": "telegram_channel_post",
            "post_id": "10",
            "text": "Program or be programmed.",
            "canonical_url": "https://t.me/rubisroundtable/10",
        }
    ]
    tg_file.write_text("".join(json.dumps(r) + "\n" for r in tg_rows), encoding="utf-8")

    sources = {
        "persona_id": "dejaru22",
        "sources": [
            {
                "type": "x_twitter",
                "index_file": str(x_file),
            },
            {
                "type": "telegram_channel",
                "index_file": str(tg_file),
            },
        ],
    }
    (root / "personas" / "dejaru22" / "SOURCES.json").write_text(
        json.dumps(sources), encoding="utf-8"
    )
    return root


def _write_persona_source(root: Path, persona_id: str, source_name: str, rows: list[dict]) -> Path:
    index_dir = root / "generic" / persona_id / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    index_file = index_dir / f"{source_name}.jsonl"
    normalized = []
    for i, row in enumerate(rows):
        normalized.append(
            {
                "source_id": row.get("source_id") or f"generic:{persona_id}:{source_name}:{i}",
                "persona_id": persona_id,
                "source_type": row.get("source_type") or "golden_fixture",
                "text": row["text"],
                "canonical_url": row.get("canonical_url"),
                "datetime": row.get("datetime"),
            }
        )
    index_file.write_text("".join(json.dumps(r) + "\n" for r in normalized), encoding="utf-8")

    persona_dir = root / "personas" / persona_id
    persona_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "persona_id": persona_id,
        "sources": [
            {
                "type": "golden_fixture",
                "handle": source_name,
                "storage_root": str(index_dir.parent),
                "index_file": str(index_file),
            }
        ],
    }
    (persona_dir / "SOURCES.json").write_text(json.dumps(sources), encoding="utf-8")
    return index_file


def _seed_golden_personas(tmp_path: Path) -> Path:
    root = tmp_path / "oracle-sources"
    _write_persona_source(
        root,
        "dejaru22",
        "dr22-golden",
        [
            {
                "source_id": "golden:dejaru22:programming",
                "text": "PROGRAM OR BE PROGRAMMED. Repetition becomes identity.",
            }
        ],
    )
    _write_persona_source(
        root,
        "yohami",
        "yohami-golden",
        [
            {
                "source_id": "golden:yohami:flirt",
                "text": "Flirt is mild sexual interest, fully expressed through calm validation.",
            }
        ],
    )
    _write_persona_source(
        root,
        "newpersona",
        "newpersona-golden",
        [
            {
                "source_id": "golden:newpersona:signal",
                "text": "A new persona becomes useful when its source IDs are stable and quotes validate.",
            }
        ],
    )
    return root


def test_load_persona_records_from_sources_registry(tmp_path, monkeypatch):
    root = _seed_persona(tmp_path)
    monkeypatch.setattr(opt, "DATA_ROOT", root)

    rows = opt._load_persona_records("dejaru22")

    assert len(rows) == 3
    assert {r["source_id"] for r in rows} == {
        "dejaru22:tweet:1",
        "dejaru22:tweet:2",
        "telegram:rubisroundtable:msg:10",
    }


def test_oracle_citation_validate_tool_checks_quote_membership(tmp_path, monkeypatch):
    root = _seed_persona(tmp_path)
    monkeypatch.setattr(opt, "DATA_ROOT", root)

    result = json.loads(
        opt.oracle_citation_validate_tool(
            {
                "persona_id": "dejaru22",
                "claims": [
                    {
                        "claim": "Use discipline",
                        "source_id": "dejaru22:tweet:1",
                        "quote": "Discipline beats mood",
                    },
                    {
                        "claim": "Wrong quote",
                        "source_id": "dejaru22:tweet:2",
                        "quote": "This quote is not present",
                    },
                ],
            }
        )
    )

    assert result["success"] is False
    assert len(result["results"]) == 2
    assert result["results"][0]["valid"] is True
    assert result["results"][1]["valid"] is False


def test_oracle_persona_search_requires_query():
    result = json.loads(opt.oracle_persona_search_tool({"persona_id": "dejaru22"}))
    assert "error" in result
    assert "query" in result["error"].lower()


def test_qdrant_client_context_closes_client_on_success(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True
    client = DummyClient()
    monkeypatch.setattr(opt, "_get_client", lambda: client)

    with opt._qdrant_client_context() as yielded:
        assert yielded is client
        assert client.closed is False

    assert client.closed is True


def test_qdrant_client_context_closes_client_on_exception(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True
    client = DummyClient()
    monkeypatch.setattr(opt, "_get_client", lambda: client)

    with pytest.raises(RuntimeError, match="boom"):
        with opt._qdrant_client_context():
            raise RuntimeError("boom")

    assert client.closed is True


def test_search_persona_closes_client_after_query(monkeypatch):
    closed = {"value": False}

    class DummyClient:
        def collection_exists(self, name):
            return True

        def query_points(self, **kwargs):
            point = type(
                "Point",
                (),
                {
                    "score": 0.9,
                    "payload": {
                        "source_id": "dejaru22:tweet:1",
                        "source_type": "x_post",
                        "datetime": None,
                        "canonical_url": "https://x.com/DejaRu22/status/1",
                        "text": "Discipline beats mood every day.",
                    },
                },
            )
            return type("Result", (), {"points": [point()]})()

        def close(self):
            closed["value"] = True

    class DummyModels:
        class Prefetch:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FusionQuery:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
    monkeypatch.setattr(opt, "_get_client", lambda: DummyClient())
    monkeypatch.setattr(opt, "_embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr(opt, "_sparse_encode", lambda text: {"sparse": text})
    monkeypatch.setattr(opt, "models", DummyModels)

    result = opt._search_persona("dejaru22", "discipline", 1, "rrf")

    assert result["success"] is True
    assert result["citations"][0]["source_id"] == "dejaru22:tweet:1"
    assert closed["value"] is True


def test_search_persona_dedupes_duplicate_source_ids(monkeypatch):
    class DummyClient:
        def collection_exists(self, name):
            return True

        def query_points(self, **kwargs):
            def point(source_id, score, text):
                return type(
                    "Point",
                    (),
                    {
                        "score": score,
                        "payload": {
                            "source_id": source_id,
                            "source_type": "x_post",
                            "datetime": None,
                            "canonical_url": f"https://example.test/{source_id}",
                            "text": text,
                        },
                    },
                )()

            return type(
                "Result",
                (),
                {
                    "points": [
                        point("dejaru22:tweet:1", 0.9, "first copy wins"),
                        point("dejaru22:tweet:1", 0.4, "duplicate should be dropped"),
                        point("dejaru22:tweet:2", 0.3, "second unique source"),
                    ]
                },
            )()

        def close(self):
            pass

    class DummyModels:
        class Prefetch:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FusionQuery:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

    monkeypatch.setattr(opt, "_get_client", lambda: DummyClient())
    monkeypatch.setattr(opt, "_embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr(opt, "_sparse_encode", lambda text: {"sparse": text})
    monkeypatch.setattr(opt, "models", DummyModels)

    result = opt._search_persona("dejaru22", "discipline", 3, "rrf")

    assert result["success"] is True
    assert [c["source_id"] for c in result["citations"]] == [
        "dejaru22:tweet:1",
        "dejaru22:tweet:2",
    ]
    assert [c["rank"] for c in result["citations"]] == [1, 2]


def test_oracle_persona_get_source_tool_returns_full_record(tmp_path, monkeypatch):
    root = _seed_persona(tmp_path)
    monkeypatch.setattr(opt, "DATA_ROOT", root)

    result = json.loads(
        opt.oracle_persona_get_source_tool(
            {"persona_id": "dejaru22", "source_id": "dejaru22:tweet:1"}
        )
    )

    assert result["success"] is True
    assert result["persona_id"] == "dejaru22"
    assert result["source"]["source_id"] == "dejaru22:tweet:1"
    assert result["source"]["text"] == "Discipline beats mood every day."
    assert result["source"]["canonical_url"] == "https://x.com/DejaRu22/status/1"


def test_oracle_persona_get_source_tool_reports_missing_source_id(tmp_path, monkeypatch):
    root = _seed_persona(tmp_path)
    monkeypatch.setattr(opt, "DATA_ROOT", root)

    result = json.loads(
        opt.oracle_persona_get_source_tool(
            {"persona_id": "dejaru22", "source_id": "dejaru22:tweet:missing"}
        )
    )

    assert result["success"] is False
    assert result["source_id"] == "dejaru22:tweet:missing"
    assert "not found" in result["error"].lower()


@pytest.mark.parametrize("bad_persona_id", ["../dejaru22", "/tmp/dejaru22", "deja/ru22", "deja\\ru22"])
def test_oracle_persona_get_source_tool_rejects_unsafe_persona_id(tmp_path, monkeypatch, bad_persona_id):
    root = _seed_persona(tmp_path)
    monkeypatch.setattr(opt, "DATA_ROOT", root)

    result = json.loads(
        opt.oracle_persona_get_source_tool(
            {"persona_id": bad_persona_id, "source_id": "dejaru22:tweet:1"}
        )
    )

    assert result["success"] is False
    assert "invalid persona_id" in result["error"].lower()


def test_load_persona_records_rejects_unsafe_persona_id(tmp_path, monkeypatch):
    root = _seed_persona(tmp_path)
    monkeypatch.setattr(opt, "DATA_ROOT", root)

    with pytest.raises(ValueError, match="invalid persona_id"):
        opt._load_persona_records("../dejaru22")


def test_collection_name_distinguishes_hyphen_and_underscore_persona_ids():
    assert opt._collection_name("foo-bar") != opt._collection_name("foo_bar")
    assert opt._collection_name("foo-bar") == "oracle_foo_dbar"
    assert opt._collection_name("foo_bar") == "oracle_foo_ubar"


def test_load_persona_records_allows_safe_hyphenated_persona_id(tmp_path, monkeypatch):
    root = tmp_path / "oracle-sources"
    _write_persona_source(
        root,
        "foo-bar",
        "hyphenated",
        [{"source_id": "golden:foo-bar:1", "text": "Hyphenated persona IDs are safe."}],
    )
    monkeypatch.setattr(opt, "DATA_ROOT", root)

    rows = opt._load_persona_records("foo-bar")

    assert rows[0]["source_id"] == "golden:foo-bar:1"


@pytest.mark.parametrize(
    "tool_call",
    [
        lambda: opt.oracle_persona_sync_tool({"persona_id": "../dejaru22"}),
        lambda: opt.oracle_persona_search_tool({"persona_id": "../dejaru22", "query": "discipline"}),
        lambda: opt.oracle_citation_validate_tool(
            {"persona_id": "../dejaru22", "claims": [{"source_id": "x", "quote": "y"}]}
        ),
    ],
)
def test_public_oracle_tools_reject_unsafe_persona_id(tool_call):
    result = json.loads(tool_call())

    assert result["success"] is False
    assert "invalid persona_id" in result["error"].lower()


def test_load_persona_records_rejects_index_file_outside_data_root(tmp_path, monkeypatch):
    root = tmp_path / "oracle-sources"
    outside = tmp_path / "outside.jsonl"
    outside.write_text(
        json.dumps({"source_id": "outside:1", "persona_id": "escape", "text": "outside"}) + "\n",
        encoding="utf-8",
    )
    persona_dir = root / "personas" / "escape"
    persona_dir.mkdir(parents=True, exist_ok=True)
    (persona_dir / "SOURCES.json").write_text(
        json.dumps({"persona_id": "escape", "sources": [{"type": "escape", "index_file": str(outside)}]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(opt, "DATA_ROOT", root)

    with pytest.raises(ValueError, match="outside ORACLE_DATA_ROOT"):
        opt._load_persona_records("escape")


def test_oracle_persona_search_auto_syncs_missing_collection(monkeypatch):
    calls = {"search": 0, "sync": 0}

    def fake_search(**kwargs):
        calls["search"] += 1
        if calls["search"] == 1:
            return {
                "success": False,
                "error": "Collection 'oracle_dejaru22' does not exist. Run oracle_persona_sync first.",
                "persona_id": kwargs["persona_id"],
                "query": kwargs["query"],
            }
        return {
            "success": True,
            "persona_id": kwargs["persona_id"],
            "query": kwargs["query"],
            "fusion": kwargs["fusion"],
            "top_k": kwargs["top_k"],
            "citations": [],
        }

    def fake_sync(**kwargs):
        calls["sync"] += 1
        return {"success": True, "persona_id": kwargs["persona_id"]}

    monkeypatch.setattr(opt, "_check_oracle_requirements", lambda: True)
    monkeypatch.setattr(opt, "_search_persona", fake_search)
    monkeypatch.setattr(opt, "_sync_persona", fake_sync)

    result = json.loads(
        opt.oracle_persona_search_tool(
            {"persona_id": "dejaru22", "query": "discipline", "top_k": 3, "fusion": "rrf"}
        )
    )

    assert result["success"] is True
    assert calls == {"search": 2, "sync": 1}


def test_direct_import_aliases_match_tool_functions(monkeypatch):
    monkeypatch.setattr(opt, "oracle_persona_search_tool", lambda args: json.dumps({"ok": args["query"]}))
    monkeypatch.setattr(opt, "oracle_persona_sync_tool", lambda args: json.dumps({"sync": True}))
    monkeypatch.setattr(opt, "oracle_citation_validate_tool", lambda args: json.dumps({"valid": True}))
    monkeypatch.setattr(opt, "oracle_persona_get_source_tool", lambda args: json.dumps({"source_id": args["source_id"]}))

    assert json.loads(opt.oracle_persona_search({"query": "discipline"})) == {"ok": "discipline"}
    assert json.loads(opt.oracle_persona_sync({})) == {"sync": True}
    assert json.loads(opt.oracle_citation_validate({"claims": [{"source_id": "x", "quote": "y"}]})) == {"valid": True}
    assert json.loads(opt.oracle_persona_get_source({"source_id": "x"})) == {"source_id": "x"}


@pytest.mark.parametrize(
    ("persona_id", "source_id", "quote"),
    [
        ("dejaru22", "golden:dejaru22:programming", "PROGRAM OR BE PROGRAMMED"),
        ("yohami", "golden:yohami:flirt", "calm validation"),
        ("newpersona", "golden:newpersona:signal", "source IDs are stable"),
    ],
)
def test_golden_persona_fixtures_validate_core_quotes(tmp_path, monkeypatch, persona_id, source_id, quote):
    root = _seed_golden_personas(tmp_path)
    monkeypatch.setattr(opt, "DATA_ROOT", root)

    source_result = json.loads(
        opt.oracle_persona_get_source_tool({"persona_id": persona_id, "source_id": source_id})
    )
    validation_result = json.loads(
        opt.oracle_citation_validate_tool(
            {"persona_id": persona_id, "claims": [{"source_id": source_id, "quote": quote}]}
        )
    )

    assert source_result["success"] is True
    assert quote in source_result["source"]["text"]
    assert validation_result["success"] is True


LIVE_GOLDEN = os.environ.get("ORACLE_RUN_GOLDEN_TESTS") == "1"


@pytest.mark.skipif(not LIVE_GOLDEN, reason="set ORACLE_RUN_GOLDEN_TESTS=1 to run live Oracle golden retrieval tests")
@pytest.mark.parametrize(
    ("persona_id", "query", "expected_source_id", "expected_quote"),
    [
        ("dejaru22", "Program or be programmed", "telegram:rubisroundtable:msg:1111", "PROGRAM OR BE PROGRAMMED."),
        ("yohami", "flirt validation attraction", "xitter:yohami:tweet:2047291656496402541", "Flirt is not communicating your attration"),
    ],
)
def test_live_golden_retrieval_returns_expected_sources(persona_id, query, expected_source_id, expected_quote):
    if not opt._check_oracle_requirements():
        pytest.skip("Oracle Qdrant/sentence-transformers dependencies are not available in this test environment")

    search_result = json.loads(
        opt.oracle_persona_search_tool(
            {"persona_id": persona_id, "query": query, "top_k": 5, "fusion": "rrf"}
        )
    )

    assert search_result["success"] is True
    assert expected_source_id in [c["source_id"] for c in search_result["citations"]]

    validation_result = json.loads(
        opt.oracle_citation_validate_tool(
            {
                "persona_id": persona_id,
                "claims": [{"source_id": expected_source_id, "quote": expected_quote}],
            }
        )
    )
    assert validation_result["success"] is True
