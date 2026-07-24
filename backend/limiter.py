"""Per-IP token bucket for the API.

Every /api/* call fans out to chess.com and lichess, so a client that hammers
this app hammers them too, under my User-Agent. The bucket starts full at
`burst` tokens and refills at `per_minute / 60` per second. One page load
firing a handful of parallel requests spends burst and passes untouched, while
a script in a loop settles down to the sustained rate.

State is per process, which suits a single instance. Across replicas each one
grants its own allowance, so the real ceiling is the limit times the replica
count.
"""

import time
from dataclasses import dataclass

# buckets are cheap, but a scan of every key on every request is not, so the
# idle ones are only swept out once the table gets big
PRUNE_THRESHOLD = 10_000


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    def __init__(self, per_minute: float, burst: int):
        if per_minute <= 0 or burst <= 0:
            raise ValueError("per_minute and burst must both be positive")
        self.per_minute = per_minute
        self.burst = burst
        self._rate = per_minute / 60
        self._buckets: dict[str, _Bucket] = {}

    def take(self, key: str) -> tuple[bool, int, float]:
        """Spend one token for `key`.

        Returns (allowed, tokens left, seconds until the next one).
        """
        now = time.monotonic()
        bucket = self._buckets.get(key)
        if bucket is None:
            self._prune(now)
            bucket = self._buckets[key] = _Bucket(float(self.burst), now)
        else:
            bucket.tokens = min(self.burst, bucket.tokens + (now - bucket.updated) * self._rate)
            bucket.updated = now

        if bucket.tokens < 1:
            return False, 0, (1 - bucket.tokens) / self._rate

        bucket.tokens -= 1
        return True, int(bucket.tokens), 0.0

    def _prune(self, now: float) -> None:
        """Forget buckets idle long enough to have refilled.

        A full bucket is indistinguishable from one that never existed, so
        dropping it costs a caller nothing.
        """
        if len(self._buckets) < PRUNE_THRESHOLD:
            return
        idle_to_full = self.burst / self._rate
        self._buckets = {
            key: bucket for key, bucket in self._buckets.items()
            if now - bucket.updated < idle_to_full
        }
