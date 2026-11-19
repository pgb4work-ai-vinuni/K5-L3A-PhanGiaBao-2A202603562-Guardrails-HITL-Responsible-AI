"""
Checkpoint 3 — Monitoring & alert.

Đếm số request / bị chặn / dính rate limit / lỗi LLM rồi so với ngưỡng.
Vượt ngưỡng thì tạo Alert (thực tế sẽ bắn sang Slack/PagerDuty, ở lab chỉ
ghi vào metrics.json + in ra terminal).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    error_rate_threshold: float = 0.2  # LLM lỗi nhiều -> có thể hết quota / sai key
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0
    output_redactions: int = 0  # số câu trả lời bị output guardrail che bớt
    llm_errors: int = 0
    blocked_by_layer: dict = field(default_factory=dict)

    def record(self, *, blocked: bool, layer: str | None, error: bool = False):
        """Gọi sau mỗi request để cập nhật bộ đếm."""
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
            key = layer or "unknown"
            self.blocked_by_layer[key] = self.blocked_by_layer.get(key, 0) + 1
        if layer == "rate_limiter":
            self.rate_limit_hits += 1
        if layer == "output_guardrail":
            self.output_redactions += 1
        if error:
            self.llm_errors += 1

    def check_metrics(self) -> list[Alert]:
        """Tính lại các tỉ lệ, trả về list Alert đang vượt ngưỡng.

        Mỗi lần gọi là tính lại từ đầu (không cộng dồn), để gọi nhiều lần
        cũng không bị alert trùng lặp.
        """
        new_alerts: list[Alert] = []

        if self.total_requests:
            block_rate = self.blocked_requests / self.total_requests
            if block_rate > self.block_rate_threshold:
                new_alerts.append(
                    Alert(
                        metric="block_rate",
                        value=block_rate,
                        threshold=self.block_rate_threshold,
                        message=(
                            f"High block rate {block_rate:.0%} "
                            f"(> {self.block_rate_threshold:.0%}) — possible attack."
                        ),
                    )
                )

        if self.rate_limit_hits > self.rate_limit_hit_threshold:
            new_alerts.append(
                Alert(
                    metric="rate_limit_hits",
                    value=self.rate_limit_hits,
                    threshold=self.rate_limit_hit_threshold,
                    message=(
                        f"Rate-limit hits {self.rate_limit_hits} "
                        f"(> {self.rate_limit_hit_threshold}) — flooding suspected."
                    ),
                )
            )

        if self.judge_checks:
            judge_fail_rate = self.judge_fails / self.judge_checks
            if judge_fail_rate > self.judge_fail_rate_threshold:
                new_alerts.append(
                    Alert(
                        metric="judge_fail_rate",
                        value=judge_fail_rate,
                        threshold=self.judge_fail_rate_threshold,
                        message=(
                            f"Judge fail rate {judge_fail_rate:.0%} "
                            f"(> {self.judge_fail_rate_threshold:.0%})."
                        ),
                    )
                )

        if self.total_requests:
            error_rate = self.llm_errors / self.total_requests
            if error_rate > self.error_rate_threshold:
                new_alerts.append(
                    Alert(
                        metric="llm_error_rate",
                        value=error_rate,
                        threshold=self.error_rate_threshold,
                        message=(
                            f"LLM error rate {error_rate:.0%} — check API key / quota / network."
                        ),
                    )
                )

        self.alerts = new_alerts
        return new_alerts

    def export_json(self, filepath: str | None = None):
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        self.check_metrics()
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": round(block_rate, 4),
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "output_redactions": self.output_redactions,
            "llm_errors": self.llm_errors,
            "blocked_by_layer": dict(self.blocked_by_layer),
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
