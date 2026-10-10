"""Scoped SQLite memory with provenance, event ordering and deletion tombstones.

Updates are deterministic storage operations, not a semantic topic classifier.
The caller must supply a verified key/scope and reliable extraction.
"""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone

LAYERS = {"session", "cross_session", "profile", "project"}


class MemoryStore:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS memory (
          user_id TEXT, scope TEXT, layer TEXT, key TEXT, value TEXT,
          event_time REAL NOT NULL, source_ids TEXT NOT NULL,
          deleted INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY(user_id, scope, layer, key)
        );
        """)

    def close(self):
        self.db.close()

    @staticmethod
    def _time(value: str) -> float:
        from .schema import instant
        return instant(value).timestamp()

    def put(self, *, user_id: str, scope: str, layer: str, key: str,
            value: str, event_time: str, source_id: str,
            reliable: bool = True) -> str:
        if layer not in LAYERS:
            raise ValueError("unknown memory layer")
        if not all((user_id, scope, key, source_id)):
            raise ValueError("scope and provenance cannot be empty")
        if not reliable:
            return "pending_review"
        when = self._time(event_time)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            old = self.db.execute(
                "SELECT * FROM memory WHERE user_id=? AND scope=? AND layer=? AND key=?",
                (user_id, scope, layer, key)).fetchone()
            if old:
                if old["deleted"]:
                    return "blocked_by_tombstone"
                if when < old["event_time"]:
                    return "ignored_stale"
                if when == old["event_time"] and value != old["value"]:
                    return "conflict"
                sources = json.loads(old["source_ids"])
                # Old value's evidence cannot be attributed to a replacement value.
                if old["value"] != value:
                    sources = [source_id]
                    status = "updated"
                else:
                    sources = sorted(set(sources + [source_id]))
                    status = "deduplicated"
            else:
                sources, status = [source_id], "created"
            self.db.execute(
                "INSERT OR REPLACE INTO memory VALUES (?,?,?,?,?,?,?,0)",
                (user_id, scope, layer, key, value, when,
                 json.dumps(sources, ensure_ascii=False)))
            return status

    def read(self, *, user_id: str, scopes: list[str], as_of: str) -> list[dict]:
        if not scopes:
            return []
        marks = ",".join("?" for _ in scopes)
        rows = self.db.execute(
            f"SELECT * FROM memory WHERE user_id=? AND scope IN ({marks}) "
            "AND deleted=0 AND event_time<=? ORDER BY layer,key",
            [user_id, *scopes, self._time(as_of)])
        return [{"layer": r["layer"], "key": r["key"], "value": r["value"],
                 "source_ids": json.loads(r["source_ids"])} for r in rows]

    def delete(self, *, user_id: str, scope: str, layer: str, key: str,
               event_time: str, approved: bool) -> str:
        if not approved:
            return "pending_review"
        if layer not in LAYERS:
            raise ValueError("unknown memory layer")
        when = self._time(event_time)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            old = self.db.execute(
                "SELECT * FROM memory WHERE user_id=? AND scope=? AND layer=? AND key=?",
                (user_id, scope, layer, key)).fetchone()
            if old and when < old["event_time"]:
                return "ignored_stale"
            self.db.execute("INSERT OR REPLACE INTO memory VALUES (?,?,?,?,?,?,?,1)",
                            (user_id, scope, layer, key, "", when, "[]"))
        return "deleted"
