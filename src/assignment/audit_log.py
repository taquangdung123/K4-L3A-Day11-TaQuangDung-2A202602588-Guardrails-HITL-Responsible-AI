"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}
        self._latest_by_user: dict[str, str] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store request metadata and return its correlation ID."""
        correlation_id = request_id or uuid.uuid4().hex
        self._open[correlation_id] = time.perf_counter()
        self._latest_by_user[user_id] = correlation_id
        self.logs.append({
            "request_id": correlation_id,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "status": "started",
        })
        return correlation_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Finish a request entry with its decision and elapsed time."""
        correlation_id = request_id or self._latest_by_user.pop(user_id, user_id)
        if request_id and self._latest_by_user.get(user_id) == request_id:
            self._latest_by_user.pop(user_id, None)
        started = self._open.pop(correlation_id, None)
        latency_ms = round((time.perf_counter() - started) * 1000, 3) if started else 0.0
        entry = next(
            (row for row in reversed(self.logs)
             if row.get("request_id") == correlation_id and row.get("status") == "started"),
            None,
        )
        if entry is None:
            entry = {
                "request_id": correlation_id,
                "user_id": user_id,
                "input": None,
                "started_at": utc_now_iso(),
            }
            self.logs.append(entry)
        entry.update({
            "output": text,
            "blocked": bool(blocked),
            "layer": layer,
            "latency_ms": latency_ms,
            "completed_at": utc_now_iso(),
            "status": "completed",
        })
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
