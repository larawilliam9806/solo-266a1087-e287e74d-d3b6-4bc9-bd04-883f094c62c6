"""Thickness-P95 feasibility and batch bounds.

A coil may be described by a summary ``(n, min, p95, max)`` or by its complete
ordered list of readings.  The module never materialises a summary as a list of
``n`` readings: every per-coil count is derived arithmetically from the four
summary numbers.  A real detail CSV is of course already a list of readings.

P95 is defined as the ``ceil(n * 0.95)``-th reading in ascending order, with
integer arithmetic (``(95*n + 99) // 100``) so that floating point round-off
can never change an index.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence


# ---------------------------------------------------------------------------
# Basic definitions
# ---------------------------------------------------------------------------

def p95_index(n: int) -> int:
    """1-based index of P95 in an ascending list of ``n`` readings."""
    if n <= 0:
        raise ValueError("point count must be positive")
    return (95 * n + 99) // 100


def summary_feasible(n: int, mn: int, p: int, mx: int) -> bool:
    """Can a coil with summary ``(n, mn, p, mx)`` exist at all?

    Requirements:
      * ``mn <= p <= mx``;
      * at least one reading equals ``mn`` and at least one equals ``mx``;
      * the ``ceil(0.95*n)``-th ascending reading equals ``p``.

    For example, with two points P95 is the maximum, so ``p`` must equal
    ``mx``.  With one point all three values must be identical.
    """
    if n <= 0:
        return False
    if not (0 <= mn <= p <= mx):
        return False

    k = p95_index(n)

    # a = # readings below p, c = # readings above p, b = # readings equal p.
    #   a <= k-1, c <= n-k, b >= 1.
    # Min/max presence forces a >= 1 when mn < p and c >= 1 when mx > p.
    a_min = 1 if mn < p else 0
    c_min = 1 if mx > p else 0

    if a_min > k - 1:
        return False
    if c_min > n - k:
        return False
    if a_min + c_min > n - 1:
        return False
    return True


def max_count_le_summary(n: int, mn: int, p: int, mx: int, t: int) -> int:
    """Maximum number of readings ``<= t`` over every feasible reading list.

    Only integer arithmetic on the four summary values is used.
    """
    if not summary_feasible(n, mn, p, mx):
        raise ValueError("infeasible summary")

    k = p95_index(n)
    c_min = 1 if mx > p else 0       # at least one reading above p (the max)
    if t < mn:
        return 0
    if t < p:
        # All "below p" readings can be set to mn; b and c readings exceed t.
        # Maximise a, keeping b >= 1 and c >= c_min.
        return min(k - 1, n - 1 - c_min)
    if t == p:
        # b readings equal p; c readings exceed p, except when p is the max.
        return n if mx == p else n - 1
    if t < mx:
        # Every reading except the forced maximum can be <= t.
        return n - 1
    return n


def max_count_ge_summary(n: int, mn: int, p: int, mx: int, t: int) -> int:
    """Maximum number of readings ``>= t`` over every feasible reading list."""
    if not summary_feasible(n, mn, p, mx):
        raise ValueError("infeasible summary")

    k = p95_index(n)
    a_min = 1 if mn < p else 0
    if t <= mn:
        return n
    if t <= p - 1:
        # One "below p" reading must be the minimum, which is below t.
        return n - 1
    if t == p:
        return n if mn == p else n - 1
    if t <= mx:
        # Only readings above p can reach t.  Maximise their count, while
        # retaining one reading equal to p (b >= 1) and the required minimum.
        return min(n - k, n - 1 - a_min)
    return 0


# ---------------------------------------------------------------------------
# Coil snapshot used for a determination
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoilSnapshot:
    coil_id: int
    coil_no: str
    n: int
    mn: int
    p: int
    mx: int
    detail_id: Optional[int] = None
    summary_id: Optional[int] = None
    readings: Optional[Sequence[int]] = None
    # readings are sorted ascending when supplied

    @property
    def has_detail(self) -> bool:
        return self.readings is not None

    @property
    def kind(self) -> str:
        return "detail" if self.has_detail else "summary"

    def count_le(self, t: int) -> int:
        if self.readings is not None:
            return bisect_right(self.readings, t)
        return max_count_le_summary(self.n, self.mn, self.p, self.mx, t)

    def count_ge(self, t: int) -> int:
        if self.readings is not None:
            return len(self.readings) - bisect_left(self.readings, t)
        return max_count_ge_summary(self.n, self.mn, self.p, self.mx, t)


def readings_stats(readings: Sequence[int]) -> tuple[int, int, int, int]:
    """Return ``(n, min, p95, max)`` for a non-empty sequence of readings."""
    vals = sorted(int(x) for x in readings)
    if not vals:
        raise ValueError("no readings supplied")
    n = len(vals)
    return n, vals[0], vals[p95_index(n) - 1], vals[-1]


# ---------------------------------------------------------------------------
# Batch bounds
# ---------------------------------------------------------------------------

MUST_EXCEED = "must_exceed"
MUST_NOT_EXCEED = "must_not_exceed"
NEEDS_DETAIL = "needs_detail"

VERDICT_LABELS = {
    MUST_EXCEED: "必超限",
    MUST_NOT_EXCEED: "必不超限",
    NEEDS_DETAIL: "待补明细",
}


@dataclass(frozen=True)
class BatchBounds:
    total_points: int
    p95_index: int
    min_possible: int
    max_possible: int
    detail_coils: int
    summary_coils: int
    per_coil: tuple[dict, ...] = field(default_factory=tuple)

    def verdict(self, limit: int) -> str:
        if self.min_possible > limit:
            return MUST_EXCEED
        if self.max_possible <= limit:
            return MUST_NOT_EXCEED
        return NEEDS_DETAIL


def batch_bounds(coils: Iterable[CoilSnapshot]) -> BatchBounds:
    """Exact range of the batch-wide P95 over all consistent readings.

    For a threshold ``t``:

    * the batch P95 can be ``<= t`` iff at least ``k`` readings can be ``<= t``;
    * the batch P95 can be ``>= t`` iff at least ``N-k+1`` readings can be
      ``>= t``.

    Both predicates are monotone in ``t`` and the per-coil maxima are
    independent, so summing the closed-form per-coil counts is exact.  No
    summary is ever expanded into individual readings.
    """
    coil_list = tuple(coils)
    if not coil_list:
        raise ValueError("batch has no coils")

    for c in coil_list:
        if c.n <= 0:
            raise ValueError(f"coil {c.coil_no} has no points")
        if c.readings is None and not summary_feasible(c.n, c.mn, c.p, c.mx):
            raise ValueError(f"coil {c.coil_no} has an infeasible summary")

    total = sum(c.n for c in coil_list)
    k = p95_index(total)
    low = min(c.mn for c in coil_list)
    high = max(c.mx for c in coil_list)

    # Smallest threshold at which k readings can be <= threshold.
    lo, hi = low, high
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(c.count_le(mid) for c in coil_list) >= k:
            hi = mid
        else:
            lo = mid + 1
    min_possible = lo

    # Largest threshold at which N-k+1 readings can be >= threshold.
    need_ge = total - k + 1
    lo, hi = low, high
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(c.count_ge(mid) for c in coil_list) >= need_ge:
            lo = mid
        else:
            hi = mid - 1
    max_possible = lo

    per_coil = tuple(
        {
            "coil_id": c.coil_id,
            "coil_no": c.coil_no,
            "kind": c.kind,
            "n": c.n,
            "min_um": c.mn,
            "p95_um": c.p,
            "max_um": c.mx,
            "summary_id": c.summary_id,
            "detail_id": c.detail_id,
        }
        for c in coil_list
    )

    return BatchBounds(
        total_points=total,
        p95_index=k,
        min_possible=min_possible,
        max_possible=max_possible,
        detail_coils=sum(1 for c in coil_list if c.has_detail),
        summary_coils=sum(1 for c in coil_list if not c.has_detail),
        per_coil=per_coil,
    )
