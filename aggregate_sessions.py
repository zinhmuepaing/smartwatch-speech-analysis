"""
aggregate_sessions.py

Parses all existing _analysis.txt files produced by Final Analysis Code.py,
builds a structured per-session pandas DataFrame, performs longitudinal trend
analysis per student, and generates student_report.txt and student_report.pdf.

Run: python aggregate_sessions.py
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
from matplotlib.backends.backend_pdf import PdfPages

# Use a CJK-capable font so student names render correctly in the PDF
_CJK_FONTS = ["Microsoft YaHei", "SimSun", "Malgun Gothic", "MS Gothic"]
for _f in _CJK_FONTS:
    if any(font.name == _f for font in _fm.fontManager.ttflist):
        matplotlib.rcParams["font.family"] = _f
        break
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIRS = [
    BASE_DIR / "FITS_ Parents Data",
    BASE_DIR / "New Dataset",
]

# Known device-ID → student name mapping (from New Dataset folder names)
DEVICE_TO_NAME = {
    "862177071030541": "陳奕揚",
    "862177071031192": "謝知霖",
    "862177071034204": "Unknown",
    "862177071036407": "陳敬謙",
    "862177071070679": "何厚智",
}

# Slope magnitude below which a trend is labelled "stable"
STABLE_SLOPE_THRESHOLD = 0.1

# Z-score multipliers used to derive personalized thresholds.
# At Z = 1.5 (one-tailed), ~6.68% of normally-distributed values exceed the
# threshold by chance (93rd percentile). Z = 2.0 → ~2.28%; Z = 1.96 → ~2.5%.
PITCH_Z_THRESHOLD = 1.5
VOLUME_Z_THRESHOLD = 1.5


# =============================================================================
# PARSING
# =============================================================================

# Matches the chunk header line, e.g. "=== 20251217_140741_chunk_1.wav ==="
_CHUNK_HEADER = re.compile(r"^=== .+ ===")

_PITCH_STAT  = re.compile(r"Pitch mean/std:\s*([\d.]+)/([\d.]+)")
_RMS_STAT    = re.compile(r"RMS mean/std:\s*([\d.]+)/([\d.]+)")
_KW_COUNT    = re.compile(r"Polite word count:\s*(\d+)")

# A tuple entry in an anomaly block looks like: "    ('0:00:02.592000', 0.065, 1.547)"
_ANOMALY_ENTRY = re.compile(r"^\s+\('[\d:. ]+',")


def parse_analysis_txt(txt_path: Path) -> list[dict]:
    """
    Parse a _analysis.txt file.

    Returns a list of dicts, one per chunk, with keys:
        pitch_mean, pitch_std, rms_mean, rms_std,
        keyword_count, pitch_alarm_count, volume_alarm_count
    """
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    chunks = []
    current: dict | None = None
    in_pitch_block = False
    in_rms_block = False

    for line in lines:
        # ── new chunk header ──────────────────────────────────────────────
        if _CHUNK_HEADER.match(line):
            if current is not None:
                chunks.append(current)
            current = {
                "pitch_mean": 0.0, "pitch_std": 0.0,
                "rms_mean": 0.0,   "rms_std": 0.0,
                "keyword_count": 0,
                "pitch_alarm_count": 0,
                "volume_alarm_count": 0,
            }
            in_pitch_block = False
            in_rms_block = False
            continue

        if current is None:
            continue

        # ── anomaly block boundaries ──────────────────────────────────────
        if line.startswith("abnormal_pitch"):
            in_pitch_block = True
            in_rms_block = False
            continue
        if line.startswith("abnormal_rms"):
            in_rms_block = True
            in_pitch_block = False
            continue
        if line.strip() == "]":
            in_pitch_block = False
            in_rms_block = False
            continue

        # ── count anomaly entries ─────────────────────────────────────────
        if in_pitch_block and _ANOMALY_ENTRY.match(line):
            current["pitch_alarm_count"] += 1
        elif in_rms_block and _ANOMALY_ENTRY.match(line):
            current["volume_alarm_count"] += 1

        # ── scalar stats ──────────────────────────────────────────────────
        m = _PITCH_STAT.search(line)
        if m:
            current["pitch_mean"] = float(m.group(1))
            current["pitch_std"]  = float(m.group(2))

        m = _RMS_STAT.search(line)
        if m:
            current["rms_mean"] = float(m.group(1))
            current["rms_std"]  = float(m.group(2))

        m = _KW_COUNT.search(line)
        if m:
            current["keyword_count"] = int(m.group(1))

    if current is not None:
        chunks.append(current)

    return chunks


# =============================================================================
# SESSION DISCOVERY
# =============================================================================

_DEVICE_RE  = re.compile(r"\b(86\d{13})\b")
_SESSION_RE = re.compile(r"(\d{8})_(\d{6})")


def _extract_device_id(path: Path) -> str:
    """Find a device ID in the path string."""
    for part in reversed(path.parts):
        m = _DEVICE_RE.search(part)
        if m:
            return m.group(1)
    return "unknown"


def discover_sessions(data_dirs: list[Path]) -> list[dict]:
    """
    Walk each data directory and collect metadata for every _analysis.txt.
    Returns list of dicts with keys:
        txt_path, student_id, student_name, session_id, session_date, data_source
    """
    sessions = []
    seen: set[str] = set()  # deduplicate by (device_id, session_id)

    for data_dir in data_dirs:
        if not data_dir.exists():
            print(f"  [skip] {data_dir.name} not found")
            continue

        for txt_path in sorted(data_dir.rglob("*_analysis.txt")):
            folder = txt_path.parent.name
            m = _SESSION_RE.match(folder)
            if not m:
                continue

            date_str, time_str = m.group(1), m.group(2)
            session_id   = f"{date_str}_{time_str}"
            session_date = pd.to_datetime(date_str, format="%Y%m%d")
            student_id   = _extract_device_id(txt_path)
            key          = f"{student_id}_{session_id}"

            if key in seen:
                continue
            seen.add(key)

            sessions.append({
                "txt_path":    txt_path,
                "student_id":  student_id,
                "student_name": DEVICE_TO_NAME.get(student_id, student_id),
                "session_id":  session_id,
                "session_date": session_date,
                "data_source": data_dir.name,
            })

    return sessions


# =============================================================================
# SESSION TABLE
# =============================================================================

def build_session_table(data_dirs: list[Path]) -> pd.DataFrame:
    """
    Parse all _analysis.txt files and return a DataFrame with one row per session.
    """
    sessions = discover_sessions(data_dirs)
    print(f"Found {len(sessions)} unique sessions across {len(data_dirs)} data directories.")

    rows = []
    for s in sessions:
        chunks = parse_analysis_txt(s["txt_path"])
        if not chunks:
            print(f"  [warn] no chunks parsed from {s['txt_path'].name}")
            continue

        row = {
            "student_id":       s["student_id"],
            "student_name":     s["student_name"],
            "session_id":       s["session_id"],
            "session_date":     s["session_date"],
            "data_source":      s["data_source"],
            "chunk_count":      len(chunks),
            "keyword_count":    sum(c["keyword_count"]    for c in chunks),
            "pitch_alarm_count":  sum(c["pitch_alarm_count"]  for c in chunks),
            "volume_alarm_count": sum(c["volume_alarm_count"] for c in chunks),
            "pitch_mean":  np.mean([c["pitch_mean"] for c in chunks]),
            "pitch_std":   np.mean([c["pitch_std"]  for c in chunks]),
            "volume_mean": np.mean([c["rms_mean"]   for c in chunks]),
            "volume_std":  np.mean([c["rms_std"]    for c in chunks]),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["student_id", "session_date"]).reset_index(drop=True)
    return df


def build_chunk_table(data_dirs: list[Path]) -> pd.DataFrame:
    """
    Parse all _analysis.txt files and return a DataFrame with one row per chunk.
    Adds chunk_index and hour_index columns for hourly aggregation.
    """
    sessions = discover_sessions(data_dirs)
    rows = []
    for s in sessions:
        chunks = parse_analysis_txt(s["txt_path"])
        if not chunks:
            print(f"  [warn] no chunks parsed from {s['txt_path'].name}")
            continue
        for i, chunk in enumerate(chunks):
            row = {
                "student_id":   s["student_id"],
                "student_name": s["student_name"],
                "session_id":   s["session_id"],
                "session_date": s["session_date"],
                "data_source":  s["data_source"],
                "chunk_index":  i,
                "hour_index":   i // 4,
            }
            row.update(chunk)
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(
        ["student_id", "session_date", "chunk_index"]
    ).reset_index(drop=True)
    print(f"Found {len(df)} total chunks across all sessions.")
    return df


def build_hourly_table(chunk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate chunk-level data to hourly granularity.
    Adds cumulative_hour column for cross-session time-series analysis.
    """
    if chunk_df.empty:
        return pd.DataFrame()
    hourly = (
        chunk_df
        .groupby(
            ["student_id", "student_name", "session_id", "session_date",
             "data_source", "hour_index"],
            sort=False,
        )
        .agg(
            chunk_count=("chunk_index", "count"),
            keyword_count=("keyword_count", "sum"),
            pitch_alarm_count=("pitch_alarm_count", "sum"),
            volume_alarm_count=("volume_alarm_count", "sum"),
            pitch_mean=("pitch_mean", "mean"),
            pitch_std=("pitch_std", "mean"),
            volume_mean=("rms_mean", "mean"),
            volume_std=("rms_std", "mean"),
        )
        .reset_index()
        .sort_values(["student_id", "session_date", "session_id", "hour_index"])
        .reset_index(drop=True)
    )
    hourly["cumulative_hour"] = hourly.groupby("student_id").cumcount()
    return hourly


# =============================================================================
# STUDENT BASELINES
# =============================================================================

def compute_student_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-student baseline statistics and derive personalized thresholds.
    """
    grp = df.groupby(["student_id", "student_name"])

    baselines = grp.agg(
        total_sessions=("session_id", "count"),
        total_keywords=("keyword_count", "sum"),
        total_pitch_alarms=("pitch_alarm_count", "sum"),
        total_volume_alarms=("volume_alarm_count", "sum"),
        baseline_pitch_mean=("pitch_mean", "mean"),
        baseline_pitch_std=("pitch_std", "mean"),             # within-session (avg chunk std)
        inter_session_pitch_std=("pitch_mean", "std"),        # between-session (std of session means)
        baseline_volume_mean=("volume_mean", "mean"),
        baseline_volume_std=("volume_std", "mean"),           # within-session
        inter_session_volume_std=("volume_mean", "std"),      # between-session
    ).reset_index()

    # std() returns NaN for single-session students — render as 0.0
    baselines["inter_session_pitch_std"]  = baselines["inter_session_pitch_std"].fillna(0.0)
    baselines["inter_session_volume_std"] = baselines["inter_session_volume_std"].fillna(0.0)

    baselines["personalized_pitch_threshold"] = (
        baselines["baseline_pitch_mean"]
        + PITCH_Z_THRESHOLD * baselines["baseline_pitch_std"]
    )
    baselines["personalized_volume_threshold"] = (
        baselines["baseline_volume_mean"]
        + VOLUME_Z_THRESHOLD * baselines["baseline_volume_std"]
    )

    return baselines


# =============================================================================
# TREND ANALYSIS
# =============================================================================

def _label_trend(slope: float, metric: str) -> str:
    """
    Convert a regression slope to a human-readable trend label.
    For alarm counts: negative slope = improving.
    For keyword count: positive slope = improving.
    """
    if abs(slope) < STABLE_SLOPE_THRESHOLD:
        return "→ stable"
    if metric == "keyword_count":
        return "↑ improving" if slope > 0 else "↓ worsening"
    else:  # alarm counts — fewer is better
        return "↓ improving" if slope < 0 else "↑ worsening"


def analyze_trends(df: pd.DataFrame) -> dict:
    """
    Per student, fit a linear trend over sessions sorted by date.
    Returns dict: student_id → {metric: (slope, label)}
    """
    trends = {}
    metrics = ["keyword_count", "pitch_alarm_count", "volume_alarm_count"]

    for student_id, grp in df.groupby("student_id"):
        grp_sorted = grp.sort_values("session_date")
        n = len(grp_sorted)
        x = np.arange(n, dtype=float)
        student_trends = {}

        for metric in metrics:
            y = grp_sorted[metric].values.astype(float)
            if n < 2 or np.all(y == y[0]):
                slope = 0.0
            else:
                slope = float(np.polyfit(x, y, 1)[0])
            student_trends[metric] = (slope, _label_trend(slope, metric))

        trends[student_id] = student_trends

    return trends


def analyze_hourly_trends(hourly_df: pd.DataFrame) -> dict:
    """
    Per student, fit a linear trend over hours sorted by cumulative_hour.
    Returns dict: student_id → {metric: (slope, label)}
    """
    trends = {}
    metrics = ["keyword_count", "pitch_alarm_count", "volume_alarm_count"]

    for student_id, grp in hourly_df.groupby("student_id"):
        grp_sorted = grp.sort_values("cumulative_hour")
        n = len(grp_sorted)
        x = grp_sorted["cumulative_hour"].values.astype(float)
        student_trends = {}

        for metric in metrics:
            y = grp_sorted[metric].values.astype(float)
            if n < 2 or np.all(y == y[0]):
                slope = 0.0
            else:
                slope = float(np.polyfit(x, y, 1)[0])
            student_trends[metric] = (slope, _label_trend(slope, metric))

        trends[student_id] = student_trends

    return trends


# =============================================================================
# REPORT GENERATION
# =============================================================================

_SESSION_TABLE_HEADER = (
    f"  {'Date':<12} {'Time':<8} {'Keywords':>9} "
    f"{'Pitch Alarms':>13} {'Vol Alarms':>11} "
    f"{'Pitch Mean (Hz)':>16} {'Vol Mean':>10}"
)
_SESSION_TABLE_SEP = "  " + "-" * 83


def _student_text_section(
    student_id: str,
    student_name: str,
    sessions: pd.DataFrame,
    hourly: pd.DataFrame,
    baseline: pd.Series,
    hourly_trends: dict,
) -> str:
    lines = []
    lines.append(f"{'='*80}")
    lines.append(f"Student: {student_name}  (Device ID: {student_id})")
    lines.append(f"{'='*80}")

    date_min = sessions["session_date"].min().strftime("%Y-%m-%d")
    date_max = sessions["session_date"].max().strftime("%Y-%m-%d")
    lines.append(f"Sessions analysed : {baseline['total_sessions']}")
    lines.append(f"Date range        : {date_min}  →  {date_max}")
    lines.append("")

    lines.append("── Baseline Statistics ──────────────────────────────────────────────────────")
    lines.append(
        f"  Pitch  : mean = {baseline['baseline_pitch_mean']:>7.2f} Hz    "
        f"within-session std = {baseline['baseline_pitch_std']:>6.2f} Hz    "
        f"between-session std = {baseline['inter_session_pitch_std']:>6.2f} Hz"
    )
    lines.append(
        f"  Volume : mean = {baseline['baseline_volume_mean']:>9.6f}    "
        f"within-session std = {baseline['baseline_volume_std']:>9.6f}    "
        f"between-session std = {baseline['inter_session_volume_std']:>9.6f}"
    )
    lines.append("")
    lines.append("── Personalized Alarm Thresholds ────────────────────────────────────────────")
    lines.append(f"  Pitch  threshold : {baseline['personalized_pitch_threshold']:>7.2f} Hz  "
                 f"(= mean + {PITCH_Z_THRESHOLD}σ)")
    lines.append(f"  Volume threshold : {baseline['personalized_volume_threshold']:>9.6f}  "
                 f"(= mean + {VOLUME_Z_THRESHOLD}σ)")
    lines.append("")

    t = hourly_trends.get(student_id, {})
    lines.append("── Hourly Trends ────────────────────────────────────────────────────────────")
    kw_avg = hourly["keyword_count"].mean()
    pa_avg = hourly["pitch_alarm_count"].mean()
    va_avg = hourly["volume_alarm_count"].mean()
    kw_label  = t.get("keyword_count",    (0, "→ stable"))[1]
    pa_label  = t.get("pitch_alarm_count", (0, "→ stable"))[1]
    va_label  = t.get("volume_alarm_count", (0, "→ stable"))[1]
    lines.append(f"  Polite keyword count  : {kw_label:<14}  (avg {kw_avg:.1f} per hour)")
    lines.append(f"  Pitch alarm count     : {pa_label:<14}  (avg {pa_avg:.1f} per hour)")
    lines.append(f"  Volume alarm count    : {va_label:<14}  (avg {va_avg:.1f} per hour)")
    lines.append("")

    lines.append("── Session-by-Session Detail ────────────────────────────────────────────────")
    lines.append(_SESSION_TABLE_HEADER)
    lines.append(_SESSION_TABLE_SEP)
    for _, row in sessions.sort_values("session_date").iterrows():
        date_part = row["session_date"].strftime("%Y-%m-%d")
        time_part = row["session_id"].split("_")[1]
        lines.append(
            f"  {date_part:<12} {time_part:<8} "
            f"{int(row['keyword_count']):>9} "
            f"{int(row['pitch_alarm_count']):>13} "
            f"{int(row['volume_alarm_count']):>11} "
            f"{row['pitch_mean']:>16.2f} "
            f"{row['volume_mean']:>10.6f}"
        )
    lines.append("")

    lines.append("── Hourly Breakdown ─────────────────────────────────────────────────────────")
    hourly_header = (
        f"  {'Session':<17} {'Hour':>5} {'Keywords':>9} "
        f"{'Pitch Alarms':>13} {'Vol Alarms':>11} "
        f"{'Pitch Mean (Hz)':>16} {'Vol Mean':>10}"
    )
    hourly_sep = "  " + "-" * 86
    lines.append(hourly_header)
    lines.append(hourly_sep)
    for _, row in hourly.sort_values(["session_date", "session_id", "hour_index"]).iterrows():
        sess_label = (
            row["session_date"].strftime("%m-%d")
            + " " + row["session_id"].split("_")[1]
        )
        hour_label = f"H{int(row['hour_index'])}"
        lines.append(
            f"  {sess_label:<17} {hour_label:>5} "
            f"{int(row['keyword_count']):>9} "
            f"{int(row['pitch_alarm_count']):>13} "
            f"{int(row['volume_alarm_count']):>11} "
            f"{row['pitch_mean']:>16.2f} "
            f"{row['volume_mean']:>10.6f}"
        )
    lines.append("")
    return "\n".join(lines)


def _student_pdf_page(
    ax_kw, ax_pitch, ax_vol,
    student_name: str,
    student_id: str,
    hourly: pd.DataFrame,
) -> None:
    """Fill three axes with hourly trend charts for one student."""
    grp = hourly.sort_values("cumulative_hour").reset_index(drop=True)
    x = np.arange(len(grp))
    hour_labels = [f"H{int(row['cumulative_hour'])}" for _, row in grp.iterrows()]

    session_starts = grp.groupby("session_id")["cumulative_hour"].min().sort_values().values
    boundary_xs = [i for i, h in enumerate(grp["cumulative_hour"].values) if h in session_starts[1:]]

    def _plot(ax, y_col, colour, ylabel, title):
        y = grp[y_col].values
        ax.plot(x, y, marker="o", color=colour, linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(hour_labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=9, pad=4)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bx in boundary_xs:
            ax.axvline(x=bx - 0.5, color="steelblue", linestyle=":", linewidth=0.8, alpha=0.5)
        if len(x) >= 2:
            slope, intercept = np.polyfit(x.astype(float), y.astype(float), 1)
            ax.plot(x, slope * x + intercept, linestyle="--", color="gray",
                    linewidth=1, alpha=0.7)

    _plot(ax_kw,    "keyword_count",    "#2196F3", "Count",    "Polite Keyword Count")
    _plot(ax_pitch, "pitch_alarm_count","#F44336", "Count",    "Pitch Alarm Count")
    _plot(ax_vol,   "volume_alarm_count","#FF9800","Count",    "Volume Alarm Count")

    ax_kw.set_title(
        f"{student_name}  ({student_id})\n" + ax_kw.get_title(),
        fontsize=9, pad=4,
    )


def generate_report(
    df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    baselines: pd.DataFrame,
    hourly_trends: dict,
    txt_path: Path,
    pdf_path: Path,
) -> None:
    """Write student_report.txt and student_report.pdf."""
    txt_sections = [
        "SMARTWATCH SPEECH ANALYTICS — STUDENT SUMMARY REPORT",
        f"Generated from {len(df)} sessions across {df['student_id'].nunique()} students\n",
    ]

    with PdfPages(pdf_path) as pdf:
        for _, bl_row in baselines.sort_values("student_name").iterrows():
            sid  = bl_row["student_id"]
            name = bl_row["student_name"]
            stu_sessions = df[df["student_id"] == sid].copy()
            stu_hourly = hourly_df[hourly_df["student_id"] == sid].copy()

            # ── text section ─────────────────────────────────────────────
            txt_sections.append(
                _student_text_section(sid, name, stu_sessions, stu_hourly, bl_row, hourly_trends)
            )

            # ── PDF page ─────────────────────────────────────────────────
            fig, (ax_kw, ax_pitch, ax_vol) = plt.subplots(
                3, 1, figsize=(10, 8), constrained_layout=True
            )
            _student_pdf_page(ax_kw, ax_pitch, ax_vol, name, sid, stu_hourly)
            pdf.savefig(fig)
            plt.close(fig)

    txt_path.write_text("\n".join(txt_sections), encoding="utf-8")
    print(f"[OK] Text report  -> {txt_path}")
    print(f"[OK] PDF report   -> {pdf_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Building session table...")
    df = build_session_table(DATA_DIRS)
    print(f"  {len(df)} sessions, {df['student_id'].nunique()} unique students\n")

    csv_path = BASE_DIR / "session_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Session summary  -> {csv_path}\n")

    print("Building hourly table...")
    chunk_df = build_chunk_table(DATA_DIRS)
    hourly_df = build_hourly_table(chunk_df)

    hourly_csv_path = BASE_DIR / "session_hourly.csv"
    hourly_df.to_csv(hourly_csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Hourly summary   -> {hourly_csv_path}\n")

    print("Computing student baselines...")
    baselines = compute_student_baselines(df)

    print("Analysing hourly trends...")
    hourly_trends = analyze_hourly_trends(hourly_df)

    print("Generating reports...")
    generate_report(
        df, hourly_df, baselines, hourly_trends,
        txt_path=BASE_DIR / "student_report.txt",
        pdf_path=BASE_DIR / "student_report.pdf",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
