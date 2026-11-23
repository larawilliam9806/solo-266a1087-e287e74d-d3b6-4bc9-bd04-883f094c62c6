"""Coil thickness batch P95 checker.

Run locally:   python app.py
Docker:        docker compose up --build
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from flask import (
    Flask,
    abort,
    flash,
    g,
    redirect,
    render_template,
    request,
    url_for,
)

import qc


BASE_DIR = Path(__file__).resolve().parent


def default_db_path() -> str:
    explicit = os.environ.get("QC_DB")
    if explicit:
        return explicit
    if Path("/data").is_dir() and os.access("/data", os.W_OK):
        return "/data/qc.db"
    return str(BASE_DIR / "data" / "qc.db")


app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("QC_SECRET", "coil-p95-dev-secret"),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,
    DATABASE=default_db_path(),
)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_no    TEXT NOT NULL UNIQUE,
    note        TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS coils (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id    INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    coil_no     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    UNIQUE (batch_id, coil_no)
);

CREATE TABLE IF NOT EXISTS summaries (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    coil_id         INTEGER NOT NULL REFERENCES coils(id) ON DELETE CASCADE,
    point_count     INTEGER NOT NULL,
    min_um          INTEGER NOT NULL,
    p95_um          INTEGER NOT NULL,
    max_um          INTEGER NOT NULL,
    active          INTEGER NOT NULL DEFAULT 1,
    retired_reason  TEXT,
    retired_at      TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_summaries_coil ON summaries(coil_id, active);

CREATE TABLE IF NOT EXISTS details (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    coil_id         INTEGER NOT NULL REFERENCES coils(id) ON DELETE CASCADE,
    readings        TEXT NOT NULL,          -- JSON array of integers
    point_count     INTEGER NOT NULL,
    min_um          INTEGER NOT NULL,
    p95_um          INTEGER NOT NULL,
    max_um          INTEGER NOT NULL,
    active          INTEGER NOT NULL DEFAULT 1,
    retired_reason  TEXT,
    retired_at      TEXT,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_details_coil ON details(coil_id, active);

CREATE TABLE IF NOT EXISTS determinations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id        INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    limit_um        INTEGER NOT NULL,
    total_points    INTEGER NOT NULL,
    p95_index       INTEGER NOT NULL,
    min_possible    INTEGER NOT NULL,
    max_possible    INTEGER NOT NULL,
    verdict         TEXT NOT NULL,
    snapshot_json   TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_det_batch ON determinations(batch_id, id);

-- A rewound finished coil is made by splicing two leftover source coils.
-- Its readings are never copied: every segment points back at the original
-- uploaded detail row and the 1-based inclusive point range taken from it,
-- so the archive keeps a single physical copy of every reading.
CREATE TABLE IF NOT EXISTS finished_coils (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    coil_id       INTEGER NOT NULL UNIQUE REFERENCES coils(id) ON DELETE CASCADE,
    batch_id      INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
    point_count   INTEGER NOT NULL,
    min_um        INTEGER NOT NULL,
    p95_um        INTEGER NOT NULL,
    max_um        INTEGER NOT NULL,
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_finished_batch ON finished_coils(batch_id);

CREATE TABLE IF NOT EXISTS finished_segments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    finished_coil_id INTEGER NOT NULL REFERENCES finished_coils(id) ON DELETE CASCADE,
    segment_order    INTEGER NOT NULL,          -- 1-based splice order
    detail_id        INTEGER NOT NULL REFERENCES details(id) ON DELETE RESTRICT,
    source_coil_no   TEXT NOT NULL,             -- captured at creation time
    source_batch_no  TEXT NOT NULL,
    start_idx        INTEGER NOT NULL,          -- 1-based, inclusive, CSV order
    end_idx          INTEGER NOT NULL,
    UNIQUE (finished_coil_id, segment_order)
);
CREATE INDEX IF NOT EXISTS idx_finseg_detail ON finished_segments(detail_id);

-- One reservation per occupied range of a source detail.  Overlap is rejected
-- inside the write transaction, so every point of one detail can serve at most
-- one finished coil.  Rows live exactly as long as the finished coil itself.
CREATE TABLE IF NOT EXISTS detail_occupancies (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    detail_id        INTEGER NOT NULL REFERENCES details(id) ON DELETE RESTRICT,
    finished_coil_id INTEGER NOT NULL REFERENCES finished_coils(id) ON DELETE CASCADE,
    start_idx        INTEGER NOT NULL,
    end_idx          INTEGER NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_occ_detail ON detail_occupancies(detail_id);
"""


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        path = app.config["DATABASE"]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def ensure_schema() -> None:
    """Create tables once per process, before the first request."""
    path = app.config["DATABASE"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=15, isolation_level=None)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
    finally:
        conn.close()


def init_db() -> None:
    ensure_schema()


@app.cli.command("init-db")
def init_db_command():
    init_db()
    print("initialised", app.config["DATABASE"])


class InputError(Exception):
    """A user-facing validation problem."""


def rollback_quiet(conn: sqlite3.Connection) -> None:
    """Roll back if a transaction is still open; never mask the real error."""
    try:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
    except sqlite3.Error:
        app.logger.exception("rollback failed")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_MAX_STAT = 10_000_000  # 10 m in micrometres, far beyond any foil


def parse_int(value, label: str, minimum: int = 0, maximum: int = _MAX_STAT) -> int:
    if value is None:
        raise InputError(f"请填写{label}")
    text = str(value).strip().replace(",", "").replace("，", "")
    if not text:
        raise InputError(f"请填写{label}")
    if not re.fullmatch(r"[+-]?\d+", text):
        raise InputError(f"{label}必须是整数（微米）")
    number = int(text)
    if number < minimum or number > maximum:
        raise InputError(f"{label}超出合理范围")
    return number


def parse_point_count(value) -> int:
    return parse_int(value, "测点数", minimum=1, maximum=5_000_000)


def parse_segment_index(value, label: str, upper: int | None = None) -> int:
    """A 1-based point index inside an uploaded detail CSV."""
    number = parse_int(value, label, minimum=1, maximum=5_000_000)
    if upper is not None and number > upper:
        raise InputError(f"{label} {number} 超出原明细范围（共 {upper} 个测点）")
    return number


def normalise_coil_no(value) -> str:
    text = (value or "").strip()
    if not text:
        raise InputError("请填写卷号")
    if len(text) > 64:
        raise InputError("卷号过长")
    if any(ch in text for ch in "\r\n\t"):
        raise InputError("卷号不能包含制表符或换行")
    return text


def normalise_batch_no(value) -> str:
    text = (value or "").strip()
    if not text:
        raise InputError("请填写批号")
    if len(text) > 64:
        raise InputError("批号过长")
    if any(ch in text for ch in "\r\n\t"):
        raise InputError("批号不能包含制表符或换行")
    return text


def summary_problem(n: int, mn: int, p: int, mx: int) -> str | None:
    """Human-readable reason a summary cannot describe any coil, or None."""
    if n <= 0:
        return "测点数必须大于零"
    if not (mn <= p <= mx):
        return "必须满足 最薄值 ≤ P95 ≤ 最厚值"
    k = qc.p95_index(n)
    if k == n and p != mx:
        return (
            f"测点数 {n} 时 P95 是第 {k} 个读数，即最厚值，"
            f"因此 P95 必须等于最厚值（{mx}）"
        )
    if k == 1 and not (mn == p == mx):
        return "测点数为 1 时 P95 就是唯一读数，三项统计必须相同"
    if not qc.summary_feasible(n, mn, p, mx):
        return "测点数与三项统计无法同时成立"
    return None


def _split_fields(line: str) -> list[str]:
    if "," in line:
        parts = line.split(",")
    elif ";" in line:
        parts = line.split(";")
    elif "\t" in line:
        parts = line.split("\t")
    else:
        parts = line.split()
    return [p.strip().strip('"').strip() for p in parts if p.strip() != ""]


def parse_readings_csv(raw: bytes) -> tuple[list[int], tuple[int, int, int] | None]:
    """Parse a complete single-coil readings CSV.

    Each line holds the thickness in micrometres, optionally followed by the
    three summary statistics (``reading,min,p95,max``).  A header line and
    blank lines are tolerated.  Returns the readings and, when present, the
    repeated per-line statistics triple.
    """
    if not raw:
        raise InputError("CSV 文件为空")
    if len(raw) > 8 * 1024 * 1024:
        raise InputError("CSV 文件过大")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("gbk")
        except UnicodeDecodeError:
            raise InputError("CSV 需为 UTF-8 或 GBK 编码") from None

    readings: list[int] = []
    triple: tuple[int, int, int] | None = None
    seen_content = False
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        fields = _split_fields(stripped)
        if not fields:
            continue
        if not re.fullmatch(r"[+-]?\d+", fields[0]):
            if not seen_content:
                seen_content = True
                continue  # header
            raise InputError(f"第 {lineno} 行不是有效的整数读数")
        seen_content = True
        try:
            values = [int(f) for f in fields]
        except ValueError:
            raise InputError(f"第 {lineno} 行含非整数内容") from None
        if len(values) == 1:
            reading, stats = values[0], None
        elif len(values) == 4:
            reading, stats = values[0], tuple(values[1:4])
        else:
            raise InputError(
                f"第 {lineno} 行应为「读数」或「读数,最薄值,P95,最厚值」"
            )
        if reading < 0 or reading > _MAX_STAT:
            raise InputError(f"第 {lineno} 行读数超出合理范围")
        if stats is not None:
            for val in stats:
                if val < 0 or val > _MAX_STAT:
                    raise InputError(f"第 {lineno} 行统计值超出合理范围")
            if triple is None:
                triple = stats
            elif triple != stats:
                raise InputError(f"第 {lineno} 行的统计摘要与前面行不一致")
        readings.append(reading)

    if not readings:
        raise InputError("CSV 中没有读数")
    return readings, triple


# ---------------------------------------------------------------------------
# Data access / snapshots
# ---------------------------------------------------------------------------

def fetch_batch(conn, batch_id: int):
    row = conn.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
    if row is None:
        abort(404)
    return row


def fetch_coils(conn, batch_id: int):
    return conn.execute(
        "SELECT * FROM coils WHERE batch_id = ? ORDER BY coil_no", (batch_id,)
    ).fetchall()


def active_summary(conn, coil_id: int):
    return conn.execute(
        "SELECT * FROM summaries WHERE coil_id = ? AND active = 1 "
        "ORDER BY id DESC LIMIT 1",
        (coil_id,),
    ).fetchone()


def active_detail(conn, coil_id: int):
    return conn.execute(
        "SELECT * FROM details WHERE coil_id = ? AND active = 1 "
        "ORDER BY id DESC LIMIT 1",
        (coil_id,),
    ).fetchone()


def ensure_coil(conn, batch_id: int, coil_no: str):
    row = conn.execute(
        "SELECT * FROM coils WHERE batch_id = ? AND coil_no = ?", (batch_id, coil_no)
    ).fetchone()
    if row is not None:
        return row
    conn.execute(
        "INSERT INTO coils(batch_id, coil_no, created_at) VALUES (?, ?, ?)",
        (batch_id, coil_no, utcnow()),
    )
    return conn.execute(
        "SELECT * FROM coils WHERE batch_id = ? AND coil_no = ?", (batch_id, coil_no)
    ).fetchone()


def retire_detail(conn, detail_id: int, reason: str) -> None:
    conn.execute(
        "UPDATE details SET active = 0, retired_reason = ?, retired_at = ? "
        "WHERE id = ? AND active = 1",
        (reason, utcnow(), detail_id),
    )


def retire_summary(conn, summary_id: int, reason: str) -> None:
    conn.execute(
        "UPDATE summaries SET active = 0, retired_reason = ?, retired_at = ? "
        "WHERE id = ? AND active = 1",
        (reason, utcnow(), summary_id),
    )


def detail_summary_mismatch(detail, summary) -> str | None:
    """Why an active detail no longer agrees with the submitted summary."""
    if summary is None:
        return None
    derived = (detail["point_count"], detail["min_um"], detail["p95_um"], detail["max_um"])
    declared = (summary["point_count"], summary["min_um"], summary["p95_um"], summary["max_um"])
    labels = ("测点数", "最薄值", "P95", "最厚值")
    diffs = [
        f"{label} 明细 {d} / 摘要 {s}"
        for label, d, s in zip(labels, derived, declared)
        if d != s
    ]
    return "；".join(diffs) if diffs else None


def build_snapshot(conn, batch_id: int) -> tuple[list[qc.CoilSnapshot], list[str]]:
    """Resolve every coil to exactly one submitted summary or detail.

    The returned snapshots are what a determination is computed from; they are
    read inside the determination transaction and persisted with the result.
    """
    coils = fetch_coils(conn, batch_id)
    if not coils:
        raise InputError("该批次还没有卷")

    snapshots: list[qc.CoilSnapshot] = []
    notes: list[str] = []
    missing: list[str] = []
    blocked: list[str] = []

    for coil in coils:
        finished = fetch_finished_by_coil(conn, coil["id"])
        if finished is not None:
            # Spliced finished coils are always accounted for from the frozen
            # source ranges captured at creation time: later retirement or
            # summary changes on a source coil never rewrite this archive.
            segments = fetch_finished_segments(conn, finished["id"])
            if not segments:
                missing.append(coil["coil_no"])
                continue
            readings = tuple(sorted(assembled_finished_readings(conn, segments)))
            derived = qc.readings_stats(readings)
            archived = (
                finished["point_count"],
                finished["min_um"],
                finished["p95_um"],
                finished["max_um"],
            )
            if derived != archived:
                blocked.append(
                    f"成品卷 {coil['coil_no']} 的接带读数实算与建档摘要不一致"
                )
                continue
            summary = active_summary(conn, coil["id"])
            snapshots.append(
                qc.CoilSnapshot(
                    coil_id=coil["id"],
                    coil_no=coil["coil_no"],
                    n=archived[0],
                    mn=archived[1],
                    p=archived[2],
                    mx=archived[3],
                    detail_id=None,
                    summary_id=summary["id"] if summary else None,
                    readings=readings,
                )
            )
            continue

        summary = active_summary(conn, coil["id"])
        detail = active_detail(conn, coil["id"])
        if summary is None and detail is None:
            missing.append(coil["coil_no"])
            continue
        problem = None
        if summary is not None:
            problem = summary_problem(
                summary["point_count"],
                summary["min_um"],
                summary["p95_um"],
                summary["max_um"],
            )
        if problem:
            blocked.append(f"卷 {coil['coil_no']} 的摘要无法成立：{problem}")
            continue

        use_detail = False
        if detail is not None and summary is not None:
            mismatch = detail_summary_mismatch(detail, summary)
            stored = json.loads(detail["readings"])
            readings = tuple(sorted(int(v) for v in stored))
            stats_ok = (
                len(readings) == detail["point_count"]
                and readings
                and qc.readings_stats(readings) == (
                    detail["point_count"],
                    detail["min_um"],
                    detail["p95_um"],
                    detail["max_um"],
                )
            )
            if mismatch is None and stats_ok:
                use_detail = True
            elif mismatch is not None:
                notes.append(
                    f"卷 {coil['coil_no']} 的明细与当前摘要不符（{mismatch}），"
                    "本次按摘要核算"
                )
            else:
                notes.append(
                    f"卷 {coil['coil_no']} 的明细档案不完整或统计与读数不符，"
                    "本次退回摘要"
                )
        elif detail is not None:
            notes.append(
                f"卷 {coil['coil_no']} 只有明细、没有已提交摘要，"
                "未满足「与摘要相符」的采用条件，本次不采用明细"
            )

        if use_detail:
            snapshots.append(
                qc.CoilSnapshot(
                    coil_id=coil["id"],
                    coil_no=coil["coil_no"],
                    n=detail["point_count"],
                    mn=detail["min_um"],
                    p=detail["p95_um"],
                    mx=detail["max_um"],
                    detail_id=detail["id"],
                    summary_id=summary["id"] if summary else None,
                    readings=readings,
                )
            )
        elif summary is not None:
            snapshots.append(
                qc.CoilSnapshot(
                    coil_id=coil["id"],
                    coil_no=coil["coil_no"],
                    n=summary["point_count"],
                    mn=summary["min_um"],
                    p=summary["p95_um"],
                    mx=summary["max_um"],
                    summary_id=summary["id"],
                )
            )
        else:
            # A detail alone is not an accounting basis: it must be checked
            # against a submitted summary first.
            missing.append(coil["coil_no"])

    if blocked:
        raise InputError("；".join(blocked))
    if missing:
        raise InputError(
            "以下卷缺少可核算的依据：" + "、".join(missing)
            + "（明细只有与该卷已提交摘要相符时才可采用；"
            "仅有明细或无任何档案都不能核算）"
        )
    return snapshots, notes


def fetch_coil_views(conn, batch_id: int):
    """Rows used by the batch page: coil plus its active and retired archives."""
    coils = fetch_coils(conn, batch_id)
    views = []
    for coil in coils:
        summary = active_summary(conn, coil["id"])
        detail = active_detail(conn, coil["id"])
        mismatch = detail_summary_mismatch(detail, summary) if detail else None
        finished = fetch_finished_by_coil(conn, coil["id"])
        finished_segments = (
            fetch_finished_segments(conn, finished["id"]) if finished else []
        )
        retired_details = conn.execute(
            "SELECT * FROM details WHERE coil_id = ? AND active = 0 "
            "ORDER BY id DESC LIMIT 5",
            (coil["id"],),
        ).fetchall()
        retired_views = [
            {
                "row": rd,
                "matches_summary": summary is not None
                and detail_summary_mismatch(rd, summary) is None,
            }
            for rd in retired_details
        ]
        views.append(
            {
                "coil": coil,
                "summary": summary,
                "detail": detail,
                "mismatch": mismatch,
                "retired_details": retired_views,
                "finished": finished,
                "finished_segments": finished_segments,
            }
        )
    return views


# ---------------------------------------------------------------------------
# Rewind / spliced finished coils
# ---------------------------------------------------------------------------

def fetch_finished_by_coil(conn, coil_id: int):
    return conn.execute(
        "SELECT * FROM finished_coils WHERE coil_id = ?", (coil_id,)
    ).fetchone()


def fetch_finished_segments(conn, finished_coil_id: int):
    return conn.execute(
        "SELECT * FROM finished_segments WHERE finished_coil_id = ? "
        "ORDER BY segment_order",
        (finished_coil_id,),
    ).fetchall()


def fetch_occupancies(conn, detail_id: int):
    return conn.execute(
        "SELECT * FROM detail_occupancies WHERE detail_id = ? ORDER BY start_idx, id",
        (detail_id,),
    ).fetchall()


def available_source_details(conn, target_batch_id: int):
    """Active details of other batches, each with the coil and its summary.

    Only details that currently pass the original adoption rule (active and
    matching their coil's active summary) are offered as splice sources.
    """
    return conn.execute(
        """
        SELECT d.*, c.coil_no AS source_coil_no, c.batch_id AS source_batch_id,
               b.batch_no AS source_batch_no, s.id AS summary_id
        FROM details d
        JOIN coils c ON c.id = d.coil_id
        JOIN batches b ON b.id = c.batch_id
        LEFT JOIN summaries s ON s.id = (
            SELECT id FROM summaries WHERE coil_id = c.id AND active = 1
            ORDER BY id DESC LIMIT 1
        )
        WHERE d.active = 1 AND c.batch_id <> ?
        ORDER BY b.batch_no, c.coil_no, d.id
        """,
        (target_batch_id,),
    ).fetchall()


def load_source_detail(conn, detail_id: int, target_batch_id: int):
    """Fetch an active splice-source detail from another batch, or raise."""
    row = conn.execute(
        """
        SELECT d.*, c.coil_no AS source_coil_no, c.batch_id AS source_batch_id,
               b.batch_no AS source_batch_no
        FROM details d
        JOIN coils c ON c.id = d.coil_id
        JOIN batches b ON b.id = c.batch_id
        WHERE d.id = ?
        """,
        (detail_id,),
    ).fetchone()
    if row is None:
        raise InputError(f"原明细 #{detail_id} 不存在")
    if row["source_batch_id"] == target_batch_id:
        raise InputError("成品只能拼接其他批次的原卷明细")
    if not row["active"]:
        raise InputError(
            f"原明细 #{detail_id}（{row['source_batch_no']} / "
            f"{row['source_coil_no']}）已停用，不能作为接带来源"
        )
    return row


def segment_readings(detail_row, start_idx: int, end_idx: int) -> list[int]:
    """Take an inclusive 1-based slice of the detail's readings in CSV order."""
    stored = json.loads(detail_row["readings"])
    if end_idx > len(stored):
        raise InputError(
            f"原明细 #{detail_row['id']} 只有 {len(stored)} 个测点，"
            f"区间 {start_idx}-{end_idx} 超出范围"
        )
    return [int(v) for v in stored[start_idx - 1:end_idx]]


def assembled_finished_readings(conn, segments) -> list[int]:
    """Concatenate the frozen source ranges in splice order.

    Readings stay where they were uploaded: this function only references the
    original detail rows, never copies their JSON payload.
    """
    ordered = sorted(segments, key=lambda r: r["segment_order"])
    collected: list[int] = []
    for seg in ordered:
        detail = conn.execute(
            "SELECT * FROM details WHERE id = ?", (seg["detail_id"],)
        ).fetchone()
        if detail is None:
            raise InputError(
                f"成品接带段引用的原明细 #{seg['detail_id']} 已不存在，无法核算"
            )
        collected.extend(segment_readings(detail, seg["start_idx"], seg["end_idx"]))
    return collected


def occupancy_overlaps(conn, detail_id: int, start_idx: int, end_idx: int,
                       exclude_finished: int | None = None):
    """Return the first existing reservation overlapping [start_idx, end_idx]."""
    sql = (
        "SELECT o.*, fc.coil_id, c.coil_no "
        "FROM detail_occupancies o "
        "JOIN finished_coils fc ON fc.id = o.finished_coil_id "
        "JOIN coils c ON c.id = fc.coil_id "
        "WHERE o.detail_id = ? AND o.start_idx <= ? AND o.end_idx >= ?"
    )
    params: list = [detail_id, end_idx, start_idx]
    if exclude_finished is not None:
        sql += " AND o.finished_coil_id <> ?"
        params.append(exclude_finished)
    sql += " ORDER BY o.id LIMIT 1"
    return conn.execute(sql, params).fetchone()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    conn = get_db()
    batches = conn.execute(
        """
        SELECT b.*,
               (SELECT COUNT(*) FROM coils c WHERE c.batch_id = b.id) AS coil_count,
               (SELECT COUNT(*) FROM determinations d WHERE d.batch_id = b.id) AS check_count
        FROM batches b
        ORDER BY b.id DESC
        """
    ).fetchall()
    return render_template("index.html", batches=batches)


@app.post("/batches")
def create_batch():
    batch_no = normalise_batch_no(request.form.get("batch_no"))
    note = (request.form.get("note") or "").strip()[:200]
    conn = get_db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        existing = conn.execute(
            "SELECT id FROM batches WHERE batch_no = ?", (batch_no,)
        ).fetchone()
        if existing:
            raise InputError(f"批号 {batch_no} 已存在")
        cur = conn.execute(
            "INSERT INTO batches(batch_no, note, created_at) VALUES (?, ?, ?)",
            (batch_no, note, utcnow()),
        )
        batch_id = cur.lastrowid
        conn.execute("COMMIT")
    except Exception:
        rollback_quiet(conn)
        raise
    flash(f"批次 {batch_no} 已建立", "ok")
    return redirect(url_for("batch_detail", batch_id=batch_id))


@app.route("/batch/<int:batch_id>")
def batch_detail(batch_id: int):
    conn = get_db()
    batch = fetch_batch(conn, batch_id)
    views = fetch_coil_views(conn, batch_id)
    sources = []
    for d in available_source_details(conn, batch_id):
        sources.append(
            {
                "row": d,
                "occupancies": fetch_occupancies(conn, d["id"]),
            }
        )
    determinations = conn.execute(
        "SELECT * FROM determinations WHERE batch_id = ? ORDER BY id DESC LIMIT 20",
        (batch_id,),
    ).fetchall()
    history = []
    for d in determinations:
        try:
            snap = json.loads(d["snapshot_json"])
        except (TypeError, ValueError):
            snap = {}
        history.append(
            {
                "row": d,
                "label": qc.VERDICT_LABELS.get(d["verdict"], d["verdict"]),
                "notes": snap.get("notes", []),
            }
        )
    return render_template(
        "batch.html", batch=batch, views=views, history=history, sources=sources
    )


@app.post("/batch/<int:batch_id>/coils")
def add_coil(batch_id: int):
    coil_no = normalise_coil_no(request.form.get("coil_no"))
    conn = get_db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        fetch_batch(conn, batch_id)
        ensure_coil(conn, batch_id, coil_no)
        conn.execute("COMMIT")
    except Exception:
        rollback_quiet(conn)
        raise
    flash(f"卷 {coil_no} 已登记", "ok")
    return redirect(url_for("batch_detail", batch_id=batch_id))


@app.post("/batch/<int:batch_id>/summary")
def register_summary(batch_id: int):
    coil_no = normalise_coil_no(request.form.get("coil_no"))
    n = parse_point_count(request.form.get("point_count"))
    mn = parse_int(request.form.get("min_um"), "最薄值")
    p = parse_int(request.form.get("p95_um"), "P95")
    mx = parse_int(request.form.get("max_um"), "最厚值")
    problem = summary_problem(n, mn, p, mx)
    if problem:
        raise InputError(f"摘要无法登记：{problem}")
    if not qc.summary_feasible(n, mn, p, mx):
        raise InputError("测点数与三项统计无法同时成立，摘要无法登记")

    conn = get_db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        fetch_batch(conn, batch_id)
        coil = ensure_coil(conn, batch_id, coil_no)
        if fetch_finished_by_coil(conn, coil["id"]) is not None:
            raise InputError(
                f"卷 {coil_no} 是复卷接带成品卷，摘要由接带建档生成，不能手工登记或换交"
            )
        current = active_summary(conn, coil["id"])
        replaced = current is not None
        if current is not None:
            retire_summary(
                conn,
                current["id"],
                f"换交摘要：{current['point_count']} 点 / "
                f"{current['min_um']}-{current['p95_um']}-{current['max_um']} → "
                f"{n} 点 / {mn}-{p}-{mx}",
            )

        detail = active_detail(conn, coil["id"])
        kept = False
        mismatch = None
        if detail is not None:
            mismatch = detail_summary_mismatch(
                detail,
                {
                    "point_count": n,
                    "min_um": mn,
                    "p95_um": p,
                    "max_um": mx,
                },
            )
            if mismatch:
                retire_detail(
                    conn,
                    detail["id"],
                    f"换交摘要后明细与摘要不符：{mismatch}",
                )
            else:
                kept = True

        conn.execute(
            """
            INSERT INTO summaries(coil_id, point_count, min_um, p95_um, max_um,
                                  active, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            """,
            (coil["id"], n, mn, p, mx, utcnow()),
        )
        conn.execute("COMMIT")
    except Exception:
        rollback_quiet(conn)
        raise

    if mismatch:
        flash(
            f"卷 {coil_no} 摘要已换交；原明细与摘要不符（{mismatch}），已停用，改按新摘要核算",
            "warn",
        )
    elif kept:
        flash(f"卷 {coil_no} 摘要已换交；现有明细与摘要相符，明细继续有效", "ok")
    elif replaced:
        flash(f"卷 {coil_no} 摘要已换交，旧摘要已作废", "ok")
    else:
        flash(f"卷 {coil_no} 摘要已登记", "ok")
    return redirect(url_for("batch_detail", batch_id=batch_id))


@app.post("/batch/<int:batch_id>/detail")
def upload_detail(batch_id: int):
    coil_no = normalise_coil_no(request.form.get("coil_no"))
    upload = request.files.get("readings")
    if upload is None or not upload.filename:
        raise InputError("请选择读数 CSV 文件")
    raw = upload.read()
    readings, triple = parse_readings_csv(raw)

    derived = qc.readings_stats(readings)
    if triple is not None and tuple(derived[1:]) != tuple(triple):
        raise InputError(
            "CSV 中携带的统计与读数不一致："
            f"读数实算 最薄 {derived[1]} / P95 {derived[2]} / 最厚 {derived[3]}，"
            f"文件声明 最薄 {triple[0]} / P95 {triple[1]} / 最厚 {triple[2]}"
        )

    conn = get_db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        fetch_batch(conn, batch_id)
        coil = ensure_coil(conn, batch_id, coil_no)
        if fetch_finished_by_coil(conn, coil["id"]) is not None:
            raise InputError(
                f"卷 {coil_no} 是复卷接带成品卷，读数只保存在原卷明细中，不能再上传明细"
            )
        if active_detail(conn, coil["id"]) is not None:
            raise InputError(
                f"卷 {coil_no} 已有有效明细；如需重传，请先停用原明细"
            )
        summary = active_summary(conn, coil["id"])
        if summary is None:
            raise InputError(
                f"卷 {coil_no} 还没有已提交摘要；明细须与摘要的点数和统计相符后才能核算，"
                "请先登记摘要"
            )
        mismatch = detail_summary_mismatch(
            {
                "point_count": derived[0],
                "min_um": derived[1],
                "p95_um": derived[2],
                "max_um": derived[3],
            },
            summary,
        )
        # "点数和三项统计都与摘要相符才按明细核算": an unmatched detail is
        # archived but immediately inactive, so accounting stays on the summary.
        active = 1 if mismatch is None else 0
        reason = None if mismatch is None else f"与摘要不符：{mismatch}"
        conn.execute(
            """
            INSERT INTO details(coil_id, readings, point_count, min_um, p95_um,
                                max_um, active, retired_reason, retired_at,
                                created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                coil["id"],
                json.dumps(readings),
                derived[0],
                derived[1],
                derived[2],
                derived[3],
                active,
                reason,
                None if active else utcnow(),
                utcnow(),
            ),
        )
        conn.execute("COMMIT")
    except Exception:
        rollback_quiet(conn)
        raise

    if mismatch:
        flash(
            f"卷 {coil_no} 明细已存档但不与摘要相符（{mismatch}）："
            f"{derived[0]} 个读数，最薄 {derived[1]} / P95 {derived[2]} / "
            f"最厚 {derived[3]} μm。明细已停用，核算退回按摘要进行",
            "warn",
        )
    else:
        flash(
            f"卷 {coil_no} 明细已接收并启用：{derived[0]} 个读数，"
            f"最薄 {derived[1]} μm、P95 {derived[2]} μm、最厚 {derived[3]} μm",
            "ok",
        )
    return redirect(url_for("batch_detail", batch_id=batch_id))


@app.post("/batch/<int:batch_id>/detail/<int:detail_id>/retire")
def retire_detail_route(batch_id: int, detail_id: int):
    conn = get_db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        fetch_batch(conn, batch_id)
        row = conn.execute(
            "SELECT d.id FROM details d JOIN coils c ON c.id = d.coil_id "
            "WHERE d.id = ? AND c.batch_id = ?",
            (detail_id, batch_id),
        ).fetchone()
        if row is None:
            raise InputError("明细不存在")
        retire_detail(conn, detail_id, "人工停用")
        conn.execute("COMMIT")
    except Exception:
        rollback_quiet(conn)
        raise
    flash("明细已停用", "ok")
    return redirect(url_for("batch_detail", batch_id=batch_id))


@app.post("/batch/<int:batch_id>/detail/<int:detail_id>/activate")
def activate_detail_route(batch_id: int, detail_id: int):
    conn = get_db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        fetch_batch(conn, batch_id)
        row = conn.execute(
            "SELECT d.* FROM details d JOIN coils c ON c.id = d.coil_id "
            "WHERE d.id = ? AND c.batch_id = ?",
            (detail_id, batch_id),
        ).fetchone()
        if row is None:
            raise InputError("明细不存在")
        if active_detail(conn, row["coil_id"]) is not None:
            raise InputError("该卷已有有效明细")
        summary = active_summary(conn, row["coil_id"])
        if summary is None:
            raise InputError("该卷没有已提交摘要；明细须与摘要相符才可用于核算，不能单独启用")
        mismatch = detail_summary_mismatch(row, summary)
        if mismatch:
            raise InputError(
                f"该明细与当前摘要不符（{mismatch}），不能启用；可先作废摘要"
            )
        readings = tuple(sorted(int(v) for v in json.loads(row["readings"])))
        if (
            not readings
            or qc.readings_stats(readings)
            != (row["point_count"], row["min_um"], row["p95_um"], row["max_um"])
        ):
            raise InputError("该明细档案不完整或统计与读数不符，不能启用")
        conn.execute(
            "UPDATE details SET active = 1, retired_reason = NULL, "
            "retired_at = NULL WHERE id = ?",
            (detail_id,),
        )
        conn.execute("COMMIT")
    except Exception:
        rollback_quiet(conn)
        raise
    flash("明细已启用", "ok")
    return redirect(url_for("batch_detail", batch_id=batch_id))


@app.post("/batch/<int:batch_id>/summary/<int:summary_id>/retire")
def retire_summary_route(batch_id: int, summary_id: int):
    conn = get_db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        fetch_batch(conn, batch_id)
        row = conn.execute(
            "SELECT s.id, s.coil_id FROM summaries s JOIN coils c ON c.id = s.coil_id "
            "WHERE s.id = ? AND c.batch_id = ?",
            (summary_id, batch_id),
        ).fetchone()
        if row is None:
            raise InputError("摘要不存在")
        if fetch_finished_by_coil(conn, row["coil_id"]) is not None:
            raise InputError("成品卷的摘要由接带建档固化，不能作废")
        retire_summary(conn, summary_id, "人工作废")
        conn.execute("COMMIT")
    except Exception:
        rollback_quiet(conn)
        raise
    flash("摘要已作废", "ok")
    return redirect(url_for("batch_detail", batch_id=batch_id))


@app.post("/batch/<int:batch_id>/determine")
def determine(batch_id: int):
    limit = parse_int(request.form.get("limit_um"), "厚度上限")
    conn = get_db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        fetch_batch(conn, batch_id)
        try:
            snapshots, notes = build_snapshot(conn, batch_id)
            bounds = qc.batch_bounds(snapshots)
        except InputError:
            raise
        except ValueError as exc:
            raise InputError(str(exc)) from exc

        verdict = bounds.verdict(limit)
        # Attach the frozen splice provenance to finished-coil snapshot rows so
        # the determination record shows which original details/ranges were used.
        finished_lookup = {
            row["coil_id"]: row
            for row in conn.execute(
                "SELECT * FROM finished_coils WHERE batch_id = ?", (batch_id,)
            ).fetchall()
        }
        per_coil = []
        for coil_snap, coil_row in zip(snapshots, bounds.per_coil):
            entry = dict(coil_row)
            fin = finished_lookup.get(coil_snap.coil_id)
            if fin is not None:
                entry["kind"] = "finished"
                entry["segments"] = [
                    {
                        "segment_order": seg["segment_order"],
                        "source_batch_no": seg["source_batch_no"],
                        "source_coil_no": seg["source_coil_no"],
                        "detail_id": seg["detail_id"],
                        "start_idx": seg["start_idx"],
                        "end_idx": seg["end_idx"],
                    }
                    for seg in fetch_finished_segments(conn, fin["id"])
                ]
            per_coil.append(entry)
        snapshot = {
            "coils": per_coil,
            "notes": notes,
            "limit_um": limit,
            "computed_at": utcnow(),
            "method": (
                "逐卷按已提交摘要的闭式计数求整批 P95 的可行区间，"
                "明细卷与复卷成品卷按完整读数计数；未把摘要展开为逐点读数。"
                "成品卷读数按建档时冻结的原卷接带区间取数"
            ),
        }
        cur = conn.execute(
            """
            INSERT INTO determinations(batch_id, limit_um, total_points,
                p95_index, min_possible, max_possible, verdict, snapshot_json,
                created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                limit,
                bounds.total_points,
                bounds.p95_index,
                bounds.min_possible,
                bounds.max_possible,
                verdict,
                json.dumps(snapshot, ensure_ascii=False),
                utcnow(),
            ),
        )
        determination_id = cur.lastrowid
        conn.execute("COMMIT")
    except Exception:
        rollback_quiet(conn)
        raise
    return redirect(
        url_for("determination_detail", batch_id=batch_id, det_id=determination_id)
    )


@app.route("/batch/<int:batch_id>/determination/<int:det_id>")
def determination_detail(batch_id: int, det_id: int):
    conn = get_db()
    batch = fetch_batch(conn, batch_id)
    row = conn.execute(
        "SELECT * FROM determinations WHERE id = ? AND batch_id = ?",
        (det_id, batch_id),
    ).fetchone()
    if row is None:
        abort(404)
    snap = json.loads(row["snapshot_json"])
    detail = {
        "row": row,
        "label": qc.VERDICT_LABELS.get(row["verdict"], row["verdict"]),
        "coils": snap.get("coils", []),
        "notes": snap.get("notes", []),
        "method": snap.get("method", ""),
        "computed_at": snap.get("computed_at", row["created_at"]),
    }
    return render_template(
        "result.html", batch=batch, det=detail, limit=row["limit_um"]
    )


# ---------------------------------------------------------------------------
# Rewind splice: build a finished coil from two other-batch source details
# ---------------------------------------------------------------------------

def _parse_segment_form(ordinal: int) -> tuple[int, int, int]:
    detail_id = parse_int(
        request.form.get(f"detail_{ordinal}"), f"第 {ordinal} 段原明细号", minimum=1
    )
    start = parse_segment_index(
        request.form.get(f"start_{ordinal}"), f"第 {ordinal} 段起始序号"
    )
    end = parse_segment_index(
        request.form.get(f"end_{ordinal}"), f"第 {ordinal} 段截止序号"
    )
    if start > end:
        raise InputError(
            f"第 {ordinal} 段起始序号 {start} 不能大于截止序号 {end}（首尾都计入）"
        )
    return detail_id, start, end


@app.post("/batch/<int:batch_id>/finished")
def create_finished(batch_id: int):
    coil_no = normalise_coil_no(request.form.get("coil_no"))
    detail_id_1, start_1, end_1 = _parse_segment_form(1)
    detail_id_2, start_2, end_2 = _parse_segment_form(2)

    conn = get_db()
    conn.execute("BEGIN IMMEDIATE")
    try:
        fetch_batch(conn, batch_id)
        existing = conn.execute(
            "SELECT id FROM coils WHERE batch_id = ? AND coil_no = ?",
            (batch_id, coil_no),
        ).fetchone()
        if existing is not None:
            raise InputError(f"成品卷号 {coil_no} 已存在")

        if detail_id_1 == detail_id_2:
            raise InputError("接带需要两卷余料：两段不能选用同一份原明细")

        src_1 = load_source_detail(conn, detail_id_1, batch_id)
        src_2 = load_source_detail(conn, detail_id_2, batch_id)
        # The two details must describe two distinct source coils.
        if src_1["coil_id"] == src_2["coil_id"]:
            raise InputError("接带需要两卷余料：两段必须来自不同的原卷")

        for src, start, end in (
            (src_1, start_1, end_1),
            (src_2, start_2, end_2),
        ):
            if end > src["point_count"]:
                raise InputError(
                    f"原明细 #{src['id']}（{src['source_batch_no']} / "
                    f"{src['source_coil_no']}）只有 {src['point_count']} 个测点，"
                    f"区间 {start}-{end} 超出原明细范围"
                )
            clash = occupancy_overlaps(conn, src["id"], start, end)
            if clash is not None:
                raise InputError(
                    f"原明细 #{src['id']} 的测点 {start}-{end} 已被成品卷 "
                    f"{clash['coil_no']} 占用（已占用区间 "
                    f"{clash['start_idx']}-{clash['end_idx']}），"
                    "同一份原明细的每个测点最多用于一卷成品"
                )

        # Readings are referenced, never duplicated: splice the two frozen
        # ranges in the submitted order and derive the finished summary.
        readings = (
            segment_readings(src_1, start_1, end_1)
            + segment_readings(src_2, start_2, end_2)
        )
        n, mn, p95, mx = qc.readings_stats(readings)

        now = utcnow()
        coil_cur = conn.execute(
            "INSERT INTO coils(batch_id, coil_no, created_at) VALUES (?, ?, ?)",
            (batch_id, coil_no, now),
        )
        new_coil_id = coil_cur.lastrowid
        fin_cur = conn.execute(
            """
            INSERT INTO finished_coils(coil_id, batch_id, point_count, min_um,
                                       p95_um, max_um, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (new_coil_id, batch_id, n, mn, p95, mx, now),
        )
        finished_id = fin_cur.lastrowid

        for ordinal, src, start, end in (
            (1, src_1, start_1, end_1),
            (2, src_2, start_2, end_2),
        ):
            conn.execute(
                """
                INSERT INTO finished_segments(finished_coil_id, segment_order,
                    detail_id, source_coil_no, source_batch_no, start_idx,
                    end_idx)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    finished_id,
                    ordinal,
                    src["id"],
                    src["source_coil_no"],
                    src["source_batch_no"],
                    start,
                    end,
                ),
            )
            conn.execute(
                """
                INSERT INTO detail_occupancies(detail_id, finished_coil_id,
                    start_idx, end_idx, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (src["id"], finished_id, start, end, now),
            )

        # The finished coil participates in the existing determination flow
        # through a normal summary row holding exactly the derived stats.
        conn.execute(
            """
            INSERT INTO summaries(coil_id, point_count, min_um, p95_um, max_um,
                                  active, created_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
            """,
            (new_coil_id, n, mn, p95, mx, now),
        )
        conn.execute("COMMIT")
    except Exception:
        rollback_quiet(conn)
        raise

    flash(
        f"成品卷 {coil_no} 已建档：{n} 个测点，最薄 {mn} / P95 {p95} / "
        f"最厚 {mx} μm，接带两段的占用已同时登记",
        "ok",
    )
    return redirect(
        url_for(
            "finished_detail",
            batch_id=batch_id,
            finished_id=finished_id,
        )
    )


@app.route("/batch/<int:batch_id>/finished/<int:finished_id>")
def finished_detail(batch_id: int, finished_id: int):
    conn = get_db()
    batch = fetch_batch(conn, batch_id)
    finished = conn.execute(
        "SELECT * FROM finished_coils WHERE id = ? AND batch_id = ?",
        (finished_id, batch_id),
    ).fetchone()
    if finished is None:
        abort(404)
    coil = conn.execute(
        "SELECT * FROM coils WHERE id = ?", (finished["coil_id"],)
    ).fetchone()
    segments = []
    for seg in fetch_finished_segments(conn, finished_id):
        detail = conn.execute(
            "SELECT * FROM details WHERE id = ?", (seg["detail_id"],)
        ).fetchone()
        segments.append(
            {
                "row": seg,
                "detail": detail,
                "occupancies": fetch_occupancies(conn, seg["detail_id"]),
            }
        )
    return render_template(
        "finished.html",
        batch=batch,
        finished=finished,
        coil=coil,
        segments=segments,
    )


@app.errorhandler(InputError)
def handle_input_error(exc):
    flash(str(exc), "error")
    target = request.referrer or url_for("index")
    return redirect(target)


@app.errorhandler(413)
def too_large(exc):
    flash("上传文件超过大小限制", "error")
    target = request.referrer or url_for("index")
    return redirect(target)


@app.errorhandler(sqlite3.Error)
def database_error(exc):
    app.logger.exception("database error")
    flash("写入失败：数据库未保存本次改动，原档案保持不变", "error")
    target = request.referrer or url_for("index")
    return redirect(target)


@app.template_filter("um")
def format_um(value):
    return f"{int(value):,} μm"


if __name__ == "__main__":
    ensure_schema()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
else:
    ensure_schema()
