"""SQLite persistence for documents, chunks and evaluation runs."""
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    filetype TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    error TEXT,
    pages INTEGER DEFAULT 0,
    stats TEXT DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page INTEGER,
    modality TEXT NOT NULL,
    content TEXT NOT NULL,
    meta TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset TEXT NOT NULL,
    created_at TEXT NOT NULL,
    config TEXT NOT NULL,
    summary TEXT NOT NULL,
    details TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, db_path: Path):
        self.db_path = str(db_path)
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # Documents -----------------------------------------------------------
    def add_document(self, filename: str, stored_path: str, filetype: str) -> int:
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO documents (filename, stored_path, filetype, created_at) VALUES (?, ?, ?, ?)",
                (filename, stored_path, filetype, _now()),
            )
            return int(cur.lastrowid)

    def update_document(self, doc_id: int, **fields: Any) -> None:
        if "stats" in fields and not isinstance(fields["stats"], str):
            fields["stats"] = json.dumps(fields["stats"])
        allowed = {"status", "error", "pages", "stats"}
        cols = [k for k in fields if k in allowed]
        if not cols:
            return
        sql = "UPDATE documents SET " + ", ".join(f"{c} = ?" for c in cols) + " WHERE id = ?"
        with self._lock, self._conn() as conn:
            conn.execute(sql, [fields[c] for c in cols] + [doc_id])

    def get_document(self, doc_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return self._doc(row) if row else None

    def list_documents(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT d.*, COUNT(c.id) AS chunk_count FROM documents d
                   LEFT JOIN chunks c ON c.doc_id = d.id GROUP BY d.id ORDER BY d.id DESC"""
            ).fetchall()
        return [self._doc(r) for r in rows]

    def delete_document(self, doc_id: int) -> list[int]:
        with self._lock, self._conn() as conn:
            ids = [r["id"] for r in conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return ids

    @staticmethod
    def _doc(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["stats"] = json.loads(d.get("stats") or "{}")
        return d

    # Chunks --------------------------------------------------------------
    def add_chunks(self, doc_id: int, chunks: Iterable[dict]) -> list[int]:
        ids: list[int] = []
        with self._lock, self._conn() as conn:
            for ch in chunks:
                cur = conn.execute(
                    "INSERT INTO chunks (doc_id, page, modality, content, meta) VALUES (?, ?, ?, ?, ?)",
                    (doc_id, ch.get("page"), ch["modality"], ch["content"], json.dumps(ch.get("meta", {}))),
                )
                ids.append(int(cur.lastrowid))
        return ids

    def all_chunks(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT c.*, d.filename FROM chunks c JOIN documents d ON d.id = c.doc_id
                   WHERE d.status = 'ready' ORDER BY c.id"""
            ).fetchall()
        return [self._chunk(r) for r in rows]

    def get_chunks(self, ids: list[int]) -> dict[int, dict]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT c.*, d.filename FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id IN ({marks})",
                ids,
            ).fetchall()
        return {r["id"]: self._chunk(r) for r in rows}

    def chunks_for_document(self, doc_id: int) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT c.*, d.filename FROM chunks c JOIN documents d ON d.id = c.doc_id
                   WHERE c.doc_id = ? ORDER BY c.page, c.id""",
                (doc_id,),
            ).fetchall()
        return [self._chunk(r) for r in rows]

    @staticmethod
    def _chunk(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["meta"] = json.loads(d.get("meta") or "{}")
        return d

    # Evaluation runs -----------------------------------------------------
    def add_eval_run(self, dataset: str, config: dict, summary: dict, details: list) -> int:
        with self._lock, self._conn() as conn:
            cur = conn.execute(
                "INSERT INTO eval_runs (dataset, created_at, config, summary, details) VALUES (?, ?, ?, ?, ?)",
                (dataset, _now(), json.dumps(config), json.dumps(summary), json.dumps(details)),
            )
            return int(cur.lastrowid)

    def list_eval_runs(self, limit: int = 20) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, dataset, created_at, config, summary FROM eval_runs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {**dict(r), "config": json.loads(r["config"]), "summary": json.loads(r["summary"])} for r in rows
        ]

    def get_eval_run(self, run_id: int) -> dict | None:
        with self._conn() as conn:
            r = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
        if not r:
            return None
        return {
            **dict(r),
            "config": json.loads(r["config"]),
            "summary": json.loads(r["summary"]),
            "details": json.loads(r["details"]),
        }
