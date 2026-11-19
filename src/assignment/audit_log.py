"""
Checkpoint 3 — Audit log.

Không chặn gì cả, chỉ ghi lại: ai hỏi gì, lúc nào, bị chặn ở lớp nào,
mất bao lâu. Khi có sự cố (vd. nghi bị leak) thì mở file này ra để truy vết.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Không lưu nguyên câu dài vài nghìn ký tự vào log, cắt bớt cho dễ đọc
_MAX_TEXT_IN_LOG = 500


def default_audit_log_path() -> str:
    """<repo>/outputs/audit_log.json — đúng cả khi chạy từ trong src/."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _clip(text: str | None) -> str:
    text = text or ""
    if len(text) <= _MAX_TEXT_IN_LOG:
        return text
    return text[:_MAX_TEXT_IN_LOG] + f"... (+{len(text) - _MAX_TEXT_IN_LOG} chars)"


class AuditLogPlugin:
    """Ghi log theo cặp input/output, nối với nhau bằng request_id."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._started_at: dict[str, float] = {}  # request_id -> perf_counter lúc bắt đầu

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        """Ghi câu hỏi + thời điểm bắt đầu. Trả về request_id để dùng cho record_output."""
        request_id = request_id or uuid.uuid4().hex[:12]
        self._started_at[request_id] = time.perf_counter()
        self.logs.append({
            "event": "input",
            "request_id": request_id,
            "user_id": user_id,
            "text": _clip(text),
            "length": len(text or ""),
            "timestamp": utc_now_iso(),
        })
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
        error: str | None = None,
    ):
        """Ghi câu trả lời, có bị chặn không, lớp nào chặn, latency."""
        started = self._started_at.pop(request_id, None) if request_id else None
        latency_ms = round((time.perf_counter() - started) * 1000, 1) if started else None
        entry = {
            "event": "output",
            "request_id": request_id,
            "user_id": user_id,
            "text": _clip(text),
            "blocked": blocked,
            "layer": layer,
            "latency_ms": latency_ms,
            "timestamp": utc_now_iso(),
        }
        if error:
            entry["error"] = error
        self.logs.append(entry)

    def summary(self) -> dict:
        outputs = [e for e in self.logs if e["event"] == "output"]
        by_layer: dict[str, int] = {}
        for e in outputs:
            if e["blocked"]:
                by_layer[e["layer"] or "unknown"] = by_layer.get(e["layer"] or "unknown", 0) + 1
        return {
            "requests": len(outputs),
            "blocked": sum(e["blocked"] for e in outputs),
            "blocked_by_layer": by_layer,
            "errors": sum(1 for e in outputs if e.get("error")),
        }

    def export_json(self, filepath: str | None = None) -> str:
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": self.summary(), "entries": self.logs}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
