import json
import pytest
from aigt.schema import Turn, Query, Document
from aigt.memory import MemoryStore
from aigt.pipeline import resolve_weather, GlassesPipeline, display_check
from aigt.retrieval import HybridRetriever
from aigt.data import read_jsonl, export_datasets, visible_messages

NOW = "2026-10-10T09:01:00+08:00"


def test_multiturn_time_and_city():
    history = [Turn("user", "明天去杭州", "2026-10-09T23:59:00+08:00")]
    q = resolve_weather(history, Turn("user", "改成上海", NOW))
    assert (q.city, q.target_date) == ("上海", "2026-10-10")


def test_retrieval_excludes_future_wrong_city_and_day():
    docs = [Document(**d) for d in read_jsonl("data/demo/documents.jsonl")]
    result = HybridRetriever(docs).search(Query("天气", NOW, "上海", "2026-10-11", "weather"))
    assert [r["document"]["id"] for r in result] == ["weather-sh-11"]


def test_memory_isolation_stale_conflict_and_delete():
    m = MemoryStore()
    fields = dict(user_id="u", scope="trip", layer="project", key="budget")
    assert m.put(**fields, value="2000", event_time=NOW, source_id="a") == "created"
    assert m.put(**fields, value="3000", event_time=NOW, source_id="b") == "conflict"
    assert m.put(**fields, value="3000", event_time="2026-10-10T10:00:00+08:00", source_id="b") == "updated"
    assert m.put(**fields, value="1000", event_time=NOW, source_id="a") == "ignored_stale"
    assert m.read(user_id="other", scopes=["trip"], as_of=NOW) == []
    rows = m.read(user_id="u", scopes=["trip"], as_of="2026-10-11T10:00:00+08:00")
    assert rows[0]["source_ids"] == ["b"]
    assert m.delete(**fields, event_time="2026-10-11T10:00:00+08:00", approved=False) == "pending_review"
    assert m.delete(**fields, event_time="2026-10-11T10:00:00+08:00", approved=True) == "deleted"
    assert m.put(**fields, value="2000", event_time=NOW, source_id="a") == "blocked_by_tombstone"
    m.close()


def test_private_verifier_excluded_and_group_split(tmp_path):
    tasks = read_jsonl("data/demo/tasks.jsonl")
    export_datasets(tasks, tmp_path)
    manifest = read_jsonl(tmp_path / "manifest.jsonl")
    for group in {m["source_group"] for m in manifest}:
        assert len({m["split"] for m in manifest if m["source_group"] == group}) == 1
    for path in tmp_path.glob("*.jsonl"):
        assert "PRIVATE_SENTINEL" not in path.read_text(encoding="utf-8")
    assert all(m["role"] != "assistant" for m in read_jsonl(tmp_path / "opd.train.jsonl")[0]["messages"])


def test_future_evidence_rejected():
    task = read_jsonl("data/demo/tasks.jsonl")[0]
    task["evidence"][0]["published_at"] = "2027-01-01T00:00:00+08:00"
    with pytest.raises(ValueError, match="future evidence"):
        visible_messages(task)


def test_display_does_not_claim_chinese_word_count():
    assert display_check("带伞出门")["compliant"] is None
    assert display_check("word " * 81)["compliant"] is False
