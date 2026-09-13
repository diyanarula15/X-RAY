"""Rate limiting against the free tier. Standard library only.

The daily cap, not the per-minute one, is what shapes this harness. The agentic
loop costs 3-5 requests per situation, so a daily allowance of a few hundred
buys a few dozen situations and a default sample becomes a multi-day job. That
is why the model default is flash-lite, whose free daily allowance is roughly
5x flash's, and why `--max-requests` exists as a hard per-run budget rather than
a safety net.

The per-day counter persists to `out/judge/_quota.json` keyed by UTC date, so
two runs on the same day share one budget instead of each believing it has the
whole thing.
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone

from .paths import QUOTA, write_atomic

# The conservative reading of the free tier at the time of writing. These move,
# and Google's published table is the authority -- re-check it rather than
# trusting this file. Both are overridable by flag and by env var.
DEFAULT_RPM = 15
DEFAULT_RPD = 1000


class QuotaExhausted(Exception):
    """The run's budget is spent. Not an error: the caller stops cleanly, and
    the cache means tomorrow's run resumes exactly here."""


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class Limiter:
    def __init__(self, rpm: int = DEFAULT_RPM, rpd: int = DEFAULT_RPD,
                 max_requests: int | None = None, sleep=time.sleep):
        self.rpm, self.rpd = rpm, rpd
        self.max_requests = max_requests
        self.sleep = sleep
        self._minute: list[float] = []
        self.used_this_run = 0
        self._day, self._used_today = self._load()

    def _load(self) -> tuple[str, int]:
        try:
            d = json.loads(QUOTA.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return _today(), 0
        return (d.get("date") or _today(),
                int(d.get("used") or 0) if d.get("date") == _today() else 0)

    def _persist(self) -> None:
        write_atomic(QUOTA, json.dumps({"date": self._day,
                                        "used": self._used_today}))

    @property
    def remaining_today(self) -> int:
        return max(0, self.rpd - self._used_today)

    @property
    def remaining_this_run(self) -> int:
        if self.max_requests is None:
            return self.remaining_today
        return max(0, min(self.remaining_today,
                          self.max_requests - self.used_this_run))

    def acquire(self, n: int = 1) -> None:
        """Block until `n` requests may be made, or raise `QuotaExhausted`."""
        if _today() != self._day:
            self._day, self._used_today = _today(), 0
        if self.remaining_this_run < n:
            raise QuotaExhausted(
                f"{self.used_this_run} requests used this run, "
                f"{self._used_today}/{self.rpd} today")
        now = time.time()
        self._minute = [t for t in self._minute if now - t < 60.0]
        if len(self._minute) + n > self.rpm:
            wait = 60.0 - (now - self._minute[0]) + 0.05
            if wait > 0:
                self.sleep(wait)
            now = time.time()
            self._minute = [t for t in self._minute if now - t < 60.0]
        for _ in range(n):
            self._minute.append(time.time())
        self.used_this_run += n
        self._used_today += n
        self._persist()


def backoff_delay(attempt: int, base: float = 60.0, cap: float = 600.0) -> float:
    """Jittered exponential backoff for a 429. The jitter matters because a
    retry storm that lands on the same second just reproduces the 429."""
    return min(base * (2 ** attempt), cap) * random.uniform(0.8, 1.2)
