"""Flask-flow tests: archives, fallback, retirement and failure preservation."""

import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="qc-test-")
os.environ["QC_DB"] = os.path.join(_TMP, "test.db")

import app as appmod  # noqa: E402


class FlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        appmod.app.config.update(TESTING=True, DATABASE=os.environ["QC_DB"])
        appmod.init_db()

    def setUp(self):
        conn = self.db()
        conn.executescript(
            "DROP TRIGGER IF EXISTS fail_sum; DROP TRIGGER IF EXISTS fail_det;"
            "DELETE FROM determinations; DELETE FROM details; "
            "DELETE FROM summaries; DELETE FROM coils; DELETE FROM batches;"
        )
        conn.commit()
        conn.close()
        self.client = appmod.app.test_client()
        self.client.post("/batches", data={"batch_no": "B1"})
        self.bid = self.db().execute("SELECT id FROM batches").fetchone()["id"]

    def db(self):
        conn = sqlite3.connect(os.environ["QC_DB"])
        conn.row_factory = sqlite3.Row
        return conn

    def summary(self, coil, n, mn, p, mx):
        return self.client.post(
            f"/batch/{self.bid}/summary",
            data={"coil_no": coil, "point_count": n, "min_um": mn,
                  "p95_um": p, "max_um": mx},
            follow_redirects=True,
        )

    def detail(self, coil, raw):
        return self.client.post(
            f"/batch/{self.bid}/detail",
            content_type="multipart/form-data",
            data={"coil_no": coil, "readings": (io.BytesIO(raw), "c.csv")},
            follow_redirects=True,
        )

    def determine(self, limit):
        return self.client.post(
            f"/batch/{self.bid}/determine",
            data={"limit_um": limit}, follow_redirects=True,
        )

    def last_det(self):
        return self.db().execute(
            "SELECT * FROM determinations ORDER BY id DESC LIMIT 1"
        ).fetchone()

    # ------------------------------------------------------------------

    def test_infeasible_summary_rejected(self):
        text = self.summary("C1", 2, 100, 200, 300).get_data(as_text=True)
        self.assertIn("无法", text)
        count = self.db().execute("SELECT COUNT(*) c FROM summaries").fetchone()["c"]
        self.assertEqual(count, 0)

    def test_bounds_and_verdicts(self):
        self.summary("C1", 2, 100, 300, 300)
        self.summary("C2", 2, 200, 300, 300)
        self.determine(300)
        d = self.last_det()
        # Total 4 readings, k=4: P95 is the batch maximum, forced to 300.
        self.assertEqual((d["min_possible"], d["max_possible"]), (300, 300))
        self.assertEqual(d["verdict"], "must_not_exceed")

        self.determine(250)
        self.assertEqual(self.last_det()["verdict"], "must_exceed")

    def test_needs_detail_band(self):
        # Total 40 readings, k=38.  C1 (20 points) never exceeds 300, so the
        # low end is 300; C2's P95=500 pushes the high end to 500.
        self.summary("C1", 20, 100, 300, 300)
        self.summary("C2", 20, 100, 500, 500)
        self.determine(400)
        d = self.last_det()
        self.assertEqual((d["min_possible"], d["max_possible"]), (300, 500), dict(d))
        self.assertEqual(d["verdict"], "needs_detail", dict(d))
        self.determine(299)
        self.assertEqual(self.last_det()["verdict"], "must_exceed")
        self.determine(500)
        self.assertEqual(self.last_det()["verdict"], "must_not_exceed")

    def test_matching_detail_is_used(self):
        self.summary("C1", 2, 100, 300, 300)
        self.detail("C1", b"reading,min,p95,max\n100,100,300,300\n300,100,300,300\n")
        row = self.db().execute("SELECT * FROM details").fetchone()
        self.assertEqual(row["active"], 1)
        self.determine(300)
        snap = json.loads(self.last_det()["snapshot_json"])
        self.assertEqual({c["coil_no"]: c["kind"] for c in snap["coils"]},
                         {"C1": "detail"})

    def test_mismatched_detail_is_archived_inactive(self):
        self.summary("C1", 2, 100, 300, 300)
        text = self.detail("C1", b"100\n200\n300\n").get_data(as_text=True)
        self.assertIn("已停用", text)
        row = self.db().execute("SELECT * FROM details").fetchone()
        self.assertEqual(row["active"], 0)
        self.assertIn("与摘要不符", row["retired_reason"])
        self.determine(300)
        snap = json.loads(self.last_det()["snapshot_json"])
        self.assertEqual(snap["coils"][0]["kind"], "summary")

    def test_summary_change_retires_unmatched_detail(self):
        self.summary("C1", 2, 100, 300, 300)
        self.detail("C1", b"100\n300\n")
        text = self.summary("C1", 3, 100, 300, 300).get_data(as_text=True)
        self.assertIn("已停用", text)
        self.assertIn("改按新摘要核算", text)
        detail = self.db().execute("SELECT * FROM details").fetchone()
        self.assertEqual(detail["active"], 0)
        active = self.db().execute(
            "SELECT * FROM summaries WHERE active = 1"
        ).fetchall()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["point_count"], 3)

    def test_summary_change_keeps_matching_detail(self):
        self.summary("C1", 2, 100, 300, 300)
        self.detail("C1", b"100\n300\n")
        text = self.summary("C1", 2, 100, 300, 300).get_data(as_text=True)
        self.assertIn("明细继续有效", text)
        detail = self.db().execute("SELECT * FROM details").fetchone()
        self.assertEqual(detail["active"], 1)

    def test_detail_without_summary_refused(self):
        text = self.detail("C1", b"100\n300\n").get_data(as_text=True)
        self.assertIn("请先登记摘要", text)
        count = self.db().execute("SELECT COUNT(*) c FROM details").fetchone()["c"]
        self.assertEqual(count, 0)

    def test_history_snapshot_is_immutable(self):
        self.summary("C1", 2, 100, 300, 300)
        self.determine(300)
        first = self.last_det()
        self.summary("C1", 2, 100, 500, 500)   # change the archive afterwards
        again = self.db().execute(
            "SELECT * FROM determinations WHERE id = ?", (first["id"],)
        ).fetchone()
        self.assertEqual(again["snapshot_json"], first["snapshot_json"])
        self.assertEqual(again["min_possible"], 300)

    def test_invalid_limit_writes_nothing(self):
        self.summary("C1", 2, 100, 300, 300)
        self.determine(300)
        before = self.db().execute("SELECT COUNT(*) c FROM determinations").fetchone()["c"]
        self.determine("abc")
        after = self.db().execute("SELECT COUNT(*) c FROM determinations").fetchone()["c"]
        self.assertEqual(before, after)

    def test_write_failure_rolls_back(self):
        self.summary("C1", 2, 100, 300, 300)
        self.detail("C1", b"100\n300\n")
        conn = self.db()
        conn.execute(
            "CREATE TRIGGER fail_sum BEFORE INSERT ON summaries "
            "BEGIN SELECT RAISE(ABORT, 'boom'); END"
        )
        conn.commit()
        conn.close()
        self.summary("C1", 3, 100, 300, 300)
        conn = self.db()
        rows = conn.execute(
            "SELECT point_count, active FROM summaries ORDER BY id"
        ).fetchall()
        detail = conn.execute("SELECT active FROM details").fetchone()
        conn.close()
        self.assertEqual([tuple(r) for r in rows], [(2, 1)])
        self.assertEqual(detail["active"], 1)

    def test_determination_failure_keeps_previous(self):
        self.summary("C1", 2, 100, 300, 300)
        self.determine(300)
        conn = self.db()
        conn.execute(
            "CREATE TRIGGER fail_det BEFORE INSERT ON determinations "
            "BEGIN SELECT RAISE(ABORT, 'boom'); END"
        )
        conn.commit()
        conn.close()
        text = self.determine(999).get_data(as_text=True)
        self.assertIn("原档案保持不变", text)
        count = self.db().execute("SELECT COUNT(*) c FROM determinations").fetchone()["c"]
        self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
