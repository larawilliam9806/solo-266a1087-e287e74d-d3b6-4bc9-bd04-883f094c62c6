"""Tests for rewind-splice finished coil archives."""

import io
import json
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TMP = tempfile.mkdtemp(prefix="qc-test-")
# When the suite is discovered, test_app imports first and points QC_DB at its
# own temp DB.  Reuse that variable instead of overwriting it, so the app's
# configured connection and these checks always target the same file.  Every
# setUp wipes all rows, so no cross-module state can leak.
if "QC_DB" not in os.environ:
    os.environ["QC_DB"] = os.path.join(_TMP, "test.db")

import app as appmod  # noqa: E402
import qc  # noqa: E402

DB_PATH = os.environ["QC_DB"]
# If test_app imported before us it may have fixed the app config to its file;
# align the running app instance with the shared path explicitly.
appmod.app.config.update(DATABASE=DB_PATH)


class FinishedCoilTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        appmod.app.config.update(TESTING=True)
        appmod.init_db()

    def setUp(self):
        # Wipe every row through one connection to the shared test DB.  The
        # extra trigger drops remove fail_sum/fail_det that test_app's rollback
        # cases leave behind (DELETE FROM does not drop triggers).
        conn = sqlite3.connect(DB_PATH)
        conn.executescript(
            "DROP TRIGGER IF EXISTS fail_sum; DROP TRIGGER IF EXISTS fail_det; "
            "DROP TRIGGER IF EXISTS fail_fin; DROP TRIGGER IF EXISTS fail_occ;"
            "DELETE FROM detail_occupancies; DELETE FROM finished_segments; "
            "DELETE FROM finished_coils; DELETE FROM determinations; "
            "DELETE FROM details; DELETE FROM summaries; DELETE FROM coils; "
            "DELETE FROM batches;"
        )
        conn.commit()
        conn.close()
        self.client = appmod.app.test_client()
        # B1/B2 hold the source leftover coils; B3 is the finished-product batch.
        self.client.post("/batches", data={"batch_no": "B1"})
        self.client.post("/batches", data={"batch_no": "B2"})
        self.client.post("/batches", data={"batch_no": "B3"})
        ids = {r["batch_no"]: r["id"] for r in self.db().execute(
            "SELECT id, batch_no FROM batches").fetchall()}
        self.b1, self.b2, self.b3 = ids["B1"], ids["B2"], ids["B3"]

    def db(self):
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

    # -- helpers ----------------------------------------------------------

    def source_coil(self, batch_id, coil, readings, mn, p95, mx):
        n = len(readings)
        self.client.post(
            f"/batch/{batch_id}/summary",
            data={"coil_no": coil, "point_count": n, "min_um": mn,
                  "p95_um": p95, "max_um": mx},
        )
        csv = ("\n".join(str(v) for v in readings) + "\n").encode()
        self.client.post(
            f"/batch/{batch_id}/detail",
            content_type="multipart/form-data",
            data={"coil_no": coil, "readings": (io.BytesIO(csv), "c.csv")},
        )
        row = self.db().execute(
            "SELECT d.* FROM details d JOIN coils c ON c.id = d.coil_id "
            "WHERE c.batch_id = ? AND c.coil_no = ?",
            (batch_id, coil),
        ).fetchone()
        self.assertEqual(row["active"], 1)
        return row["id"]

    def make_finished(self, coil_no, d1, s1, e1, d2, s2, e2,
                      batch_id=None, follow=True):
        return self.client.post(
            f"/batch/{batch_id or self.b3}/finished",
            data={
                "coil_no": coil_no,
                "detail_1": d1, "start_1": s1, "end_1": e1,
                "detail_2": d2, "start_2": s2, "end_2": e2,
            },
            follow_redirects=follow,
        )

    def finished_row(self, coil_no):
        return self.db().execute(
            "SELECT fc.* FROM finished_coils fc JOIN coils c ON c.id = fc.coil_id "
            "WHERE c.coil_no = ?", (coil_no,)
        ).fetchone()

    # -- happy path --------------------------------------------------------

    def test_finished_archive_derives_summary_and_occupancy(self):
        # A: 4 points 100..400 (P95 of 4 is the max); B: 4 points 150..450.
        da = self.source_coil(self.b1, "A", [100, 200, 300, 400], 100, 400, 400)
        db_ = self.source_coil(self.b2, "B", [150, 250, 350, 450], 150, 450, 450)
        page = self.client.get(f"/batch/{self.b3}").get_data(as_text=True)
        self.assertIn("复卷接带建档", page)
        self.assertIn(f"明细 #{da}", page)

        resp = self.make_finished("F1", da, 1, 4, db_, 1, 4)
        self.assertEqual(resp.status_code, 200)

        conn = self.db()
        fin = conn.execute(
            "SELECT * FROM finished_coils JOIN coils c ON c.id = finished_coils.coil_id "
            "WHERE c.coil_no = 'F1'").fetchone()
        # Spliced readings sorted: 100,150,200,250,300,350,400,450; n=8, k=8.
        self.assertEqual((fin["point_count"], fin["min_um"],
                          fin["p95_um"], fin["max_um"]), (8, 100, 450, 450))

        segs = conn.execute(
            "SELECT * FROM finished_segments WHERE finished_coil_id = ? "
            "ORDER BY segment_order", (fin["id"],)).fetchall()
        self.assertEqual(len(segs), 2)
        self.assertEqual(
            [(s["segment_order"], s["detail_id"], s["start_idx"], s["end_idx"],
              s["source_coil_no"])
             for s in segs],
            [(1, da, 1, 4, "A"), (2, db_, 1, 4, "B")],
        )
        occ = conn.execute(
            "SELECT * FROM detail_occupancies ORDER BY detail_id").fetchall()
        self.assertEqual({(o["detail_id"], o["start_idx"], o["end_idx"]) for o in occ},
                         {(da, 1, 4), (db_, 1, 4)})
        # A normal summary row backs the existing determination flow.
        summary = conn.execute(
            "SELECT s.* FROM summaries s JOIN coils c ON c.id = s.coil_id "
            "WHERE c.coil_no = 'F1' AND s.active = 1").fetchone()
        self.assertEqual((summary["point_count"], summary["min_um"],
                          summary["p95_um"], summary["max_um"]),
                         (8, 100, 450, 450))
        # Readings are stored once: no readings payload on the finished side.
        self.assertEqual(
            conn.execute("SELECT COUNT(*) c FROM details").fetchone()["c"], 2
        )
        conn.close()

    def test_finished_detail_page_lists_sources_and_occupancy(self):
        da = self.source_coil(self.b1, "A", [100, 200, 300, 400], 100, 400, 400)
        db_ = self.source_coil(self.b2, "B", [150, 250, 350, 450], 150, 450, 450)
        self.make_finished("F1", da, 1, 3, db_, 2, 4)
        fin = self.finished_row("F1")
        text = self.client.get(
            f"/batch/{self.b3}/finished/{fin['id']}").get_data(as_text=True)
        self.assertIn("复卷成品卷 F1", text)
        self.assertIn("原明细号", text)
        self.assertIn(f"#{da}", text)
        self.assertIn("1-3", text)
        self.assertIn("2-4", text)

    def test_determination_uses_spliced_readings_with_provenance(self):
        da = self.source_coil(self.b1, "A", [100, 200, 300, 400], 100, 400, 400)
        db_ = self.source_coil(self.b2, "B", [150, 250, 350, 450], 150, 450, 450)
        self.make_finished("F1", da, 1, 4, db_, 1, 4)
        self.client.post(f"/batch/{self.b3}/determine",
                         data={"limit_um": 450}, follow_redirects=True)
        det = self.db().execute(
            "SELECT * FROM determinations ORDER BY id DESC LIMIT 1").fetchone()
        snap = json.loads(det["snapshot_json"])
        self.assertEqual(len(snap["coils"]), 1)
        coil = snap["coils"][0]
        self.assertEqual(coil["kind"], "finished")
        self.assertEqual((coil["n"], coil["min_um"], coil["p95_um"],
                          coil["max_um"]), (8, 100, 450, 450))
        self.assertEqual([s["detail_id"] for s in coil["segments"]], [da, db_])
        # Exact reading-based bounds: the only P95 (8th of 8) is 450.
        self.assertEqual((det["min_possible"], det["max_possible"]), (450, 450))

    def test_partial_ranges_are_inclusive_and_ordered(self):
        da = self.source_coil(self.b1, "A", [100, 200, 300, 400], 100, 400, 400)
        db_ = self.source_coil(self.b2, "B", [150, 250, 350, 450], 150, 450, 450)
        # Segment 1 takes A[2..3] = 200,300; segment 2 takes B[1..2] = 150,250.
        self.make_finished("F1", da, 2, 3, db_, 1, 2)
        fin = self.finished_row("F1")
        self.assertEqual(fin["point_count"], 4)  # inclusive, both ends counted
        self.assertEqual((fin["min_um"], fin["max_um"]), (150, 300))
        n = fin["point_count"]
        # Sorted: 150,200,250,300; k = (95*4+99)//100 = 4 -> P95 = 300.
        self.assertEqual(fin["p95_um"],
                         sorted([200, 300, 150, 250])[qc.p95_index(n) - 1])

    # -- occupancy rules ---------------------------------------------------

    def test_overlapping_range_rejected_and_writes_nothing(self):
        da = self.source_coil(self.b1, "A", [100, 200, 300, 400], 100, 400, 400)
        db_ = self.source_coil(self.b2, "B", [150, 250, 350, 450], 150, 450, 450)
        self.make_finished("F1", da, 1, 2, db_, 1, 2)
        before = self.db().execute(
            "SELECT COUNT(*) c FROM finished_coils").fetchone()["c"]
        # F2 reuses A points 2-3 (point 2 already taken) — must be rejected.
        text = self.make_finished("F2", da, 2, 3, db_, 3, 4).get_data(as_text=True)
        self.assertIn("已被成品卷 F1 占用", text)
        conn = self.db()
        self.assertEqual(
            conn.execute("SELECT COUNT(*) c FROM finished_coils").fetchone()["c"],
            before,
        )
        self.assertEqual(
            conn.execute("SELECT COUNT(*) c FROM coils WHERE coil_no = 'F2'")
            .fetchone()["c"], 0)
        self.assertEqual(
            conn.execute("SELECT COUNT(*) c FROM detail_occupancies").fetchone()["c"],
            2)
        conn.close()

    def test_non_overlapping_ranges_of_same_detail_allowed(self):
        da = self.source_coil(self.b1, "A", [100, 200, 300, 400], 100, 400, 400)
        db_ = self.source_coil(self.b2, "B", [150, 250, 350, 450], 150, 450, 450)
        self.make_finished("F1", da, 1, 2, db_, 1, 2)
        # Adjacent, non-overlapping ranges (3-4 vs 1-2) are fine.
        self.make_finished("F2", da, 3, 4, db_, 3, 4)
        conn = self.db()
        self.assertEqual(
            conn.execute("SELECT COUNT(*) c FROM finished_coils").fetchone()["c"], 2)
        self.assertEqual(
            conn.execute("SELECT COUNT(*) c FROM detail_occupancies").fetchone()["c"],
            4)
        conn.close()

    # -- validation ---------------------------------------------------------

    def test_retired_or_switched_source_detail_is_rejected(self):
        da = self.source_coil(self.b1, "A", [100, 200, 300, 400], 100, 400, 400)
        db_ = self.source_coil(self.b2, "B", [150, 250, 350, 450], 150, 450, 450)
        self.client.post(
            f"/batch/{self.b1}/detail/{da}/retire", follow_redirects=True)
        text = self.make_finished("F1", da, 1, 2, db_, 1, 2).get_data(as_text=True)
        self.assertIn("已停用", text)
        self.assertIsNone(self.finished_row("F1"))

    def test_duplicate_finished_coil_no_rejected(self):
        da = self.source_coil(self.b1, "A", [100, 200, 300, 400], 100, 400, 400)
        db_ = self.source_coil(self.b2, "B", [150, 250, 350, 450], 150, 450, 450)
        self.make_finished("F1", da, 1, 2, db_, 1, 2)
        text = self.make_finished("F1", da, 3, 4, db_, 3, 4).get_data(as_text=True)
        self.assertIn("成品卷号 F1 已存在", text)

    def test_index_out_of_range_and_reversed_interval_rejected(self):
        da = self.source_coil(self.b1, "A", [100, 200, 300, 400], 100, 400, 400)
        db_ = self.source_coil(self.b2, "B", [150, 250, 350, 450], 150, 450, 450)
        text = self.make_finished("F1", da, 1, 9, db_, 1, 2).get_data(as_text=True)
        self.assertIn("超出原明细范围", text)
        text = self.make_finished("F1", da, 3, 2, db_, 1, 2).get_data(as_text=True)
        self.assertIn("不能大于", text)
        self.assertIsNone(self.finished_row("F1"))

    def test_same_detail_or_same_coil_twice_rejected(self):
        da = self.source_coil(self.b1, "A", [100, 200, 300, 400], 100, 400, 400)
        db_ = self.source_coil(self.b2, "B", [150, 250, 350, 450], 150, 450, 450)
        text = self.make_finished("F1", da, 1, 2, da, 3, 4).get_data(as_text=True)
        self.assertIn("不能选用同一份原明细", text)
        # A second active detail on the same source coil is also the same coil.
        self.client.post(
            f"/batch/{self.b1}/summary",
            data={"coil_no": "A", "point_count": 4, "min_um": 100,
                  "p95_um": 400, "max_um": 400})
        text = self.make_finished("F1", da, 1, 2, da, 1, 2).get_data(as_text=True)
        self.assertTrue("不能选用同一份原明细" in text or "同一卷" in text)
        self.assertIsNone(self.finished_row("F1"))
        # Sanity: db_ is still a valid distinct source (used nowhere yet).
        self.make_finished("F1", da, 1, 2, db_, 1, 2)
        self.assertIsNotNone(self.finished_row("F1"))

    def test_same_batch_detail_cannot_be_source(self):
        # Upload a detail inside the target batch B3 itself.
        d_same = self.source_coil(self.b3, "X", [100, 200, 300, 400],
                                  100, 400, 400)
        db_ = self.source_coil(self.b2, "B", [150, 250, 350, 450],
                                150, 450, 450)
        text = self.make_finished("F1", d_same, 1, 2, db_, 1, 2).get_data(as_text=True)
        self.assertIn("其他批次", text)
        self.assertIsNone(self.finished_row("F1"))

    # -- freeze after source switch ----------------------------------------

    def test_source_switch_after_creation_keeps_finished_archive(self):
        da = self.source_coil(self.b1, "A", [100, 200, 300, 400], 100, 400, 400)
        db_ = self.source_coil(self.b2, "B", [150, 250, 350, 450], 150, 450, 450)
        self.make_finished("F1", da, 1, 4, db_, 1, 4)

        # Source coil A changes summary -> its old detail is retired ("换档").
        self.client.post(
            f"/batch/{self.b1}/summary",
            data={"coil_no": "A", "point_count": 3, "min_um": 100,
                  "p95_um": 300, "max_um": 300}, follow_redirects=True)
        retired = self.db().execute(
            "SELECT active FROM details WHERE id = ?", (da,)).fetchone()
        self.assertEqual(retired["active"], 0)

        # The finished batch still determines against the frozen original data.
        self.client.post(f"/batch/{self.b3}/determine",
                         data={"limit_um": 450}, follow_redirects=True)
        det = self.db().execute(
            "SELECT * FROM determinations ORDER BY id DESC LIMIT 1").fetchone()
        self.assertEqual(det["total_points"], 8)
        self.assertEqual((det["min_possible"], det["max_possible"]), (450, 450))
        fin = self.finished_row("F1")
        segs = self.db().execute(
            "SELECT * FROM finished_segments WHERE finished_coil_id = ? "
            "ORDER BY segment_order", (fin["id"],)).fetchall()
        self.assertEqual(segs[0]["detail_id"], da)  # still points at the original

    def test_other_batches_remain_independent(self):
        da = self.source_coil(self.b1, "A", [100, 200, 300, 400], 100, 400, 400)
        db_ = self.source_coil(self.b2, "B", [150, 250, 350, 450], 150, 450, 450)
        self.make_finished("F1", da, 1, 2, db_, 1, 2)
        # Source batches still run their own determinations exactly as before.
        self.client.post(f"/batch/{self.b1}/determine",
                         data={"limit_um": 400}, follow_redirects=True)
        d1 = self.db().execute(
            "SELECT * FROM determinations WHERE batch_id = ? ORDER BY id DESC LIMIT 1",
            (self.b1,)).fetchone()
        self.assertEqual(d1["verdict"], "must_not_exceed")

    # -- finished coil is immutable via ordinary routes --------------------

    def test_finished_coil_rejects_manual_archive_edits(self):
        da = self.source_coil(self.b1, "A", [100, 200, 300, 400], 100, 400, 400)
        db_ = self.source_coil(self.b2, "B", [150, 250, 350, 450], 150, 450, 450)
        self.make_finished("F1", da, 1, 2, db_, 1, 2)
        text = self.client.post(
            f"/batch/{self.b3}/summary",
            data={"coil_no": "F1", "point_count": 2, "min_um": 1,
                  "p95_um": 1, "max_um": 1},
            follow_redirects=True).get_data(as_text=True)
        self.assertIn("接带建档", text)
        text = self.client.post(
            f"/batch/{self.b3}/detail",
            content_type="multipart/form-data",
            data={"coil_no": "F1",
                  "readings": (io.BytesIO(b"1\n2\n"), "x.csv")},
            follow_redirects=True).get_data(as_text=True)
        self.assertIn("不能再上传明细", text)
        summary_id = self.db().execute(
            "SELECT s.id FROM summaries s JOIN coils c ON c.id = s.coil_id "
            "WHERE c.coil_no = 'F1'").fetchone()["id"]
        text = self.client.post(
            f"/batch/{self.b3}/summary/{summary_id}/retire",
            follow_redirects=True).get_data(as_text=True)
        self.assertIn("不能作废", text)
        # Archive untouched.
        self.assertEqual(
            self.db().execute(
                "SELECT active FROM summaries WHERE id = ?", (summary_id,)
            ).fetchone()["active"], 1)

    # -- atomicity ----------------------------------------------------------

    def test_write_failure_rolls_back_finished_archive_and_occupancy(self):
        da = self.source_coil(self.b1, "A", [100, 200, 300, 400], 100, 400, 400)
        db_ = self.source_coil(self.b2, "B", [150, 250, 350, 450], 150, 450, 450)
        conn = self.db()
        conn.execute(
            "CREATE TRIGGER fail_occ BEFORE INSERT ON detail_occupancies "
            "BEGIN SELECT RAISE(ABORT, 'boom'); END"
        )
        conn.commit()
        conn.close()
        text = self.make_finished("F1", da, 1, 2, db_, 1, 2).get_data(as_text=True)
        self.assertIn("原档案保持不变", text)
        conn = self.db()
        self.assertEqual(
            conn.execute("SELECT COUNT(*) c FROM finished_coils").fetchone()["c"], 0)
        self.assertEqual(
            conn.execute("SELECT COUNT(*) c FROM finished_segments").fetchone()["c"], 0)
        self.assertEqual(
            conn.execute("SELECT COUNT(*) c FROM detail_occupancies").fetchone()["c"], 0)
        self.assertEqual(
            conn.execute("SELECT COUNT(*) c FROM coils WHERE coil_no = 'F1'")
            .fetchone()["c"], 0)
        self.assertEqual(
            conn.execute("SELECT COUNT(*) c FROM summaries").fetchone()["c"], 2)
        conn.close()
        # After the trigger is dropped the same submission succeeds.
        conn = self.db()
        conn.execute("DROP TRIGGER fail_occ")
        conn.commit()
        conn.close()
        self.make_finished("F1", da, 1, 2, db_, 1, 2)
        self.assertIsNotNone(self.finished_row("F1"))


if __name__ == "__main__":
    unittest.main()
