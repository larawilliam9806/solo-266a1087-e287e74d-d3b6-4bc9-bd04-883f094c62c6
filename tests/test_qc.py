"""Exhaustive brute-force checks of the closed-form summary arithmetic."""

import itertools
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qc


def feasible_lists(n, mn, p, mx, hi=6):
    k = qc.p95_index(n) - 1
    return [
        combo
        for combo in itertools.combinations_with_replacement(range(hi + 1), n)
        if combo[0] == mn and combo[-1] == mx and combo[k] == p
    ]


class SummaryCountsHashedOut(unittest.TestCase):
    def test_feasibility_and_counts_match_enumeration(self):
        for n in range(1, 8):
            for mn in range(0, 5):
                for p in range(mn, 6):
                    for mx in range(p, 7):
                        lists = feasible_lists(n, mn, p, mx)
                        self.assertEqual(
                            bool(lists), qc.summary_feasible(n, mn, p, mx),
                            (n, mn, p, mx),
                        )
                        if not lists:
                            continue
                        for t in range(-1, 8):
                            le = max(sum(v <= t for v in r) for r in lists)
                            ge = max(sum(v >= t for v in r) for r in lists)
                            self.assertEqual(
                                qc.max_count_le_summary(n, mn, p, mx, t), le,
                                ("le", n, mn, p, mx, t),
                            )
                            self.assertEqual(
                                qc.max_count_ge_summary(n, mn, p, mx, t), ge,
                                ("ge", n, mn, p, mx, t),
                            )

    def test_batch_bounds_match_enumeration(self):
        rng = random.Random(20261123)
        for _ in range(250):
            coils, choices = [], []
            for i in range(rng.randint(1, 3)):
                n = rng.randint(1, 5)
                vals = sorted(rng.randint(0, 8) for _ in range(n))
                _, mn, p, mx = qc.readings_stats(vals)
                if rng.random() < 0.5:
                    coils.append(qc.CoilSnapshot(i, f"C{i}", n, mn, p, mx, summary_id=1))
                    choices.append(feasible_lists(n, mn, p, mx, hi=8))
                else:
                    coils.append(
                        qc.CoilSnapshot(i, f"C{i}", n, mn, p, mx,
                                        detail_id=1, readings=tuple(vals))
                    )
                    choices.append([tuple(vals)])
            bounds = qc.batch_bounds(coils)
            k = qc.p95_index(sum(c.n for c in coils)) - 1
            possible = set()
            for combo in itertools.product(*choices):
                merged = sorted(v for r in combo for v in r)
                possible.add(merged[k])
            self.assertEqual(bounds.min_possible, min(possible), bounds)
            self.assertEqual(bounds.max_possible, max(possible), bounds)

    def test_verdict_partition(self):
        bounds = qc.BatchBounds(10, 10, 300, 320, 0, 1)
        self.assertEqual(bounds.verdict(299), qc.MUST_EXCEED)
        self.assertEqual(bounds.verdict(300), qc.NEEDS_DETAIL)
        self.assertEqual(bounds.verdict(310), qc.NEEDS_DETAIL)
        self.assertEqual(bounds.verdict(320), qc.MUST_NOT_EXCEED)


if __name__ == "__main__":
    unittest.main()
