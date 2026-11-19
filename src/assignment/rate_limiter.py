"""
Checkpoint 3 — Rate limiter (sliding window, theo từng user).

Lớp này đứng ĐẦU pipeline: chặn spam / flood trước khi tốn công chạy regex
hay tốn tiền gọi LLM. Các guardrail khác không xử lý được kiểu tấn công này
(mỗi câu riêng lẻ đều "sạch", chỉ là gửi quá nhiều).
"""
from __future__ import annotations

import contextvars
import time
from collections import defaultdict, deque

from google.adk.plugins import base_plugin
from google.genai import types

# OpenAIRunner của lab luôn tạo invocation_context với user_id="student",
# nên nếu chỉ đọc từ context thì mọi người dùng bị gộp chung 1 quota.
# Pipeline set biến này trước mỗi request để rate limiter biết ai đang gọi.
current_user_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_user_id", default=None
)


class RateLimitPlugin(base_plugin.BasePlugin):
    """Mỗi user tối đa `max_requests` câu trong `window_seconds` giây gần nhất."""

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        super().__init__(name="rate_limiter")
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # user_id -> các timestamp của request ĐƯỢC CHO QUA trong cửa sổ
        self.user_windows: dict[str, deque[float]] = defaultdict(deque)
        self.blocked_count = 0
        self.total_count = 0

    def _resolve_user(self, invocation_context) -> str:
        return (
            current_user_id.get()
            or getattr(invocation_context, "user_id", None)
            or "anonymous"
        )

    def _block_response(self, message: str) -> types.Content:
        return types.Content(role="model", parts=[types.Part.from_text(text=message)])

    async def on_user_message_callback(self, *, invocation_context, user_message):
        """Content = chặn, None = cho qua."""
        self.total_count += 1
        user_id = self._resolve_user(invocation_context)
        now = time.monotonic()  # monotonic: không bị ảnh hưởng khi chỉnh giờ hệ thống
        window = self.user_windows[user_id]

        # bỏ các request đã trôi ra khỏi cửa sổ
        while window and now - window[0] >= self.window_seconds:
            window.popleft()

        if len(window) >= self.max_requests:
            self.blocked_count += 1
            retry_after = self.window_seconds - (now - window[0])
            return self._block_response(
                f"Rate limit exceeded ({self.max_requests} requests / "
                f"{self.window_seconds}s). Try again in {retry_after:.0f}s."
            )

        # request bị chặn KHÔNG được ghi vào window, nếu không user spam
        # liên tục sẽ bị khoá mãi mãi
        window.append(now)
        return None
