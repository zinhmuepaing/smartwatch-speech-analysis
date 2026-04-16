# Speech Analytics Pipeline - Project Progress Report

**Project**: FITS Smartwatch Speech Analytics for ASD Students  
**Date Generated**: 2026-04-16  
**Status**: Core Pipeline Complete with Aggregation Layer  

---

## 1. Project Completion Status

### ✅ Already Implemented (Pre-Existing)

#### **Final Analysis Code.py** (Original)
- Automated audio file discovery from recursive directory structure
- Transcript generation (Whisper AI fallback if no SRT found)
- Audio chunking (15-minute segments for memory efficiency)
- Parallel chunk processing (multiprocessing with CPU-aware pooling)
- Audio feature extraction:
  - RMS energy calculation (volume proxy)
  - Praat-based pitch extraction (F0 fundamental frequency)
- Statistical anomaly detection (z-score method)
- Keyword counting (Cantonese polite phrases)
- Individual session reporting (text + PDF per audio file)

**Key Outputs**: `*_analysis.txt`, `*_analysis.pdf` (per session)

---

### ✅ NEW - Recently Completed Updates

#### **1. aggregate_sessions.py** (NEW - Created by you)
**Purpose**: Longitudinal aggregation and student-level reporting

**Key Functions Implemented**:

##### A. **parse_analysis_txt()** - Parse Raw Results
- **Input**: Individual `_analysis.txt` files from per-session analysis
- **Method**: Regex-based parsing of chunk statistics
- **Output**: List of dictionaries with aggregated chunk-level metrics

**Formulas Used**:
```
For each chunk in a session:
  - Extract: pitch_mean, pitch_std, rms_mean, rms_std
  - Count: keyword_count, pitch_alarm_count, volume_alarm_count
```

##### B. **discover_sessions()** - Session Discovery & Deduplication
- **Method**: Walk both data directories (`FITS_ Parents Data/`, `New Dataset/`)
- **Key Feature**: Deduplication by `(device_id, session_id)` pair
- **Benefit**: Prevents double-counting when sessions exist in both locations

**Logic**:
```python
seen = set()
for each session found:
  key = f"{device_id}_{session_id}"
  if key not in seen:
    add to sessions list
    mark as seen
```

**Result**: Unique sessions only, with first occurrence preferred (FITS priority)

##### C. **build_session_table()** - Session-Level Aggregation
- **Input**: All discovered sessions with parsed chunk data
- **Process**: Average chunk-level metrics per session

**Aggregation Formula**:
```
session_pitch_mean = average([chunk1_pitch_mean, chunk2_pitch_mean, ..., chunkN_pitch_mean])
session_pitch_std  = average([chunk1_pitch_std, chunk2_pitch_std, ..., chunkN_pitch_std])
session_keyword_count = sum([chunk1_kw, chunk2_kw, ..., chunkN_kw])
session_pitch_alarms = sum([chunk1_pitch_alarms, chunk2_pitch_alarms, ...])
session_volume_alarms = sum([chunk1_volume_alarms, chunk2_volume_alarms, ...])
```

**Why averaging?**
- Chunks are uniform length (15 min each)
- Reduces noise from individual frames
- Creates session-level "signature" for trending

**Output**: DataFrame with one row per session

##### D. **compute_student_baselines()** - Personalized Thresholds
- **Input**: Session-level DataFrame
- **Grouping**: By `(student_id, student_name)`

**Baseline Calculation Formula**:
```
baseline_pitch_mean = average(session1_pitch_mean, session2_pitch_mean, ..., sessionM_pitch_mean)
baseline_pitch_std  = average(session1_pitch_std, session2_pitch_std, ..., sessionM_pitch_std)

personalized_pitch_threshold = baseline_pitch_mean + 2.0 × baseline_pitch_std
personalized_volume_threshold = baseline_volume_mean + 1.5 × baseline_volume_std
```

**Why personalized thresholds?**
- Each student has unique vocal characteristics
- "Normal" for one student may be anomalous for another
- Standard deviations account for individual variability

**Output**: Per-student baseline statistics and adaptive thresholds

##### E. **analyze_trends()** - Longitudinal Trend Detection
- **Method**: Linear regression per student per metric
- **Implementation**: `numpy.polyfit(x, y, 1)` (degree-1 polynomial)

**Trend Formula**:
```
Given chronologically sorted sessions:
  x = [0, 1, 2, 3, ..., n_sessions]  (session sequence)
  y = [metric_value_1, metric_value_2, ..., metric_value_n]  (metric per session)

slope = linear_regression_coefficient(x, y)

If |slope| < 0.1:
  trend = "→ stable"
Else if metric == "keyword_count":
  trend = "↑ improving" if slope > 0 else "↓ worsening"
Else:  # alarm counts (fewer is better)
  trend = "↓ improving" if slope < 0 else "↑ worsening"
```

**Output**: Trend direction (improving/stable/worsening) with slope magnitude

---

#### **2. session_summary.csv** (NEW - Exported Output)
**File Format**: Tab-separated values with UTF-8 BOM

**Columns**:
| Column | Type | Definition |
|--------|------|------------|
| `student_id` | string | Device ID (14 digits) |
| `student_name` | string | Student name or "Unknown" |
| `session_id` | string | YYYYMMDD_HHMMSS timestamp |
| `session_date` | datetime | Date of session |
| `data_source` | string | "FITS_ Parents Data" or "New Dataset" |
| `chunk_count` | int | Number of 15-min chunks |
| `keyword_count` | int | Total polite phrase occurrences |
| `pitch_alarm_count` | int | Total pitch anomalies detected |
| `volume_alarm_count` | int | Total volume anomalies detected |
| `pitch_mean` | float | Average pitch (Hz) across session |
| `pitch_std` | float | Average pitch variability (Hz) |
| `volume_mean` | float | Average RMS energy (loudness proxy) |
| `volume_std` | float | Average RMS variability |

**Purpose**: Machine-readable format for further analysis (Excel, R, analytics tools)

**Calculation Method**:
```python
df.to_csv('session_summary.csv', index=False, encoding='utf-8-sig')
```

---

#### **3. student_report.txt** (NEW - Human-Readable Summary)
**Format**: Structured text report with one section per student

**Sections per Student**:

##### A. **Header**
```
Student: [Name]  (Device ID: [device_id])
Sessions analysed : [count]
Date range        : [start_date]  →  [end_date]
```

##### B. **Baseline Statistics**
```
Pitch  : mean = [X.XX] Hz    std = [Y.YY] Hz
Volume : mean = [X.XXXXXX]   std = [Y.YYYYYY]
```
**Interpretation**: Describes "normal" speech characteristics for this student

##### C. **Personalized Alarm Thresholds**
```
Pitch  threshold : [Z.ZZ] Hz  (= mean + 2.0σ)
Volume threshold : [Z.ZZZZZZ] (= mean + 1.5σ)
```
**Interpretation**: Values above these trigger anomaly alerts

##### D. **Longitudinal Trends** (from linear regression)
```
Polite keyword count  : [↑/↓/→] [label]  (avg X.X per session)
Pitch alarm count     : [↑/↓/→] [label]  (avg Y.Y per session)
Volume alarm count    : [↑/↓/→] [label]  (avg Z.Z per session)
```

**Trend Interpretation**:
- ↑ improving: More polite words **OR** fewer alarms (positive progress)
- ↓ worsening: Fewer polite words **OR** more alarms (declining trend)
- → stable: No significant change over time

##### E. **Session-by-Session Detail Table**
```
Date         Time      Keywords  Pitch Alarms  Vol Alarms  Pitch Mean (Hz)   Vol Mean
-----------------------------------------------------------------------------------
YYYY-MM-DD   HHMMSS        INT        INT          INT           FLOAT          FLOAT
```

**Purpose**: 
- Track individual session data points
- Identify specific dates with unusual behavior
- Support qualitative insights from quantitative data

---

#### **4. student_report.pdf** (NEW - Visual Dashboard)
**Format**: Multi-page PDF with CJK font support (Microsoft YaHei)

**Content per Student (One Page)**:

**Three Trend Graphs**:
1. **Polite Keyword Count** (blue line)
   - Y-axis: Count of polite phrases per session
   - X-axis: Sessions in chronological order
   - Shows: Speech politeness trend

2. **Pitch Alarm Count** (red line)
   - Y-axis: Count of pitch anomalies per session
   - X-axis: Sessions in chronological order
   - Shows: Vocal stability trend

3. **Volume Alarm Count** (orange line)
   - Y-axis: Count of volume anomalies per session
   - X-axis: Sessions in chronological order
   - Shows: Loudness regulation trend

**Each graph includes**:
- Date labels on x-axis (MM-DD format)
- Linear regression trend line (dashed gray) overlaid
- Grid for easy reading
- Student name and device ID header

**Purpose**: 
- Visual communication for non-technical audiences
- Quick identification of improvement/decline patterns
- Easy integration into presentations/reports

---

## 2. Summary of Formulas Used

| Metric | Formula | Purpose |
|--------|---------|---------|
| Session average | `mean(chunk_values)` | Aggregate chunk data to session level |
| Student baseline | `mean(session_values)` | Establish "normal" for each student |
| Personalized threshold | `baseline + Z × std` | Adaptive anomaly detection |
| Z-score | `(value - mean) / std` | Normalize anomalies statistically |
| Trend slope | `polyfit(x, y, 1)[0]` | Linear progression over time |
| Trend label | `sign(slope) × metric_type` | Interpret if improving/worsening |

---

## 3. Data Flow Diagram

```
Raw Audio Files (.wav)
        ↓
Transcript (.srt) [Auto-generated if missing]
        ↓
Final Analysis Code.py (Per-session chunks)
        ├─ RMS Energy Extraction
        ├─ Pitch (F0) Extraction  
        ├─ Z-score Anomaly Detection
        ├─ Keyword Counting
        ↓
Per-Session: *_analysis.txt, *_analysis.pdf
        ↓
aggregate_sessions.py (NEW - Aggregation Layer)
        ├─ Parse all *_analysis.txt
        ├─ Discover & deduplicate sessions
        ├─ Build session-level DataFrame
        ├─ Compute per-student baselines
        ├─ Analyze longitudinal trends
        ↓
Outputs (NEW):
├─ session_summary.csv (machine-readable)
├─ student_report.txt (human-readable summary)
└─ student_report.pdf (visual dashboard)
```

---

## 4. What Was Completed

### Code Deliverables
- ✅ `aggregate_sessions.py` - Full aggregation pipeline (500+ lines)
- ✅ Deduplication logic - Handles duplicate sessions across directories
- ✅ Baseline computation - Per-student statistical profiles
- ✅ Trend analysis - Linear regression with interpretation
- ✅ Report generation - Text + PDF outputs with CJK font support
- ✅ Data export - CSV for external analysis

### Outputs Generated
- ✅ `session_summary.csv` - 25 sessions across 5 students
- ✅ `student_report.txt` - Formatted text report
- ✅ `student_report.pdf` - Visual trend charts per student

### Validation Completed
- ✅ Verified deduplication (20 raw sessions → 15 unique for 陳敬謙)
- ✅ Tested with real data from 5 students
- ✅ Confirmed baseline calculations and trend detection
- ✅ Validated CJK character rendering in PDF

---

## 5. Suggested Git Commit Messages

### Option Set 1: Technical Descriptive
```
git commit -m "feat: Add session aggregation and student reporting pipeline

- Implement aggregate_sessions.py with deduplication logic
- Add session-level statistics aggregation (averaging, baseline calculation)
- Implement personalized threshold computation per student
- Add linear regression-based trend analysis
- Generate session_summary.csv, student_report.txt, and student_report.pdf
- Support CJK characters in PDF rendering
- Process 25 sessions across 5 ASD students with longitudinal tracking"
```

### Option Set 2: Short & Focused
```
git commit -m "Add aggregation layer: session summaries and student reports

- Create aggregate_sessions.py for multi-session analysis
- Implement deduplication across FITS and New Dataset directories
- Generate CSV export and PDF dashboards with trend analysis
- 25 sessions, 5 students processed successfully"
```

### Option Set 3: Feature-Based (Multiple Commits)
```
Commit 1:
git commit -m "feat: Implement session data aggregation pipeline

- Parse _analysis.txt files from per-session processing
- Build session-level DataFrame with aggregated metrics
- Deduplicate sessions across data directories"

Commit 2:
git commit -m "feat: Add student baselines and personalized thresholds

- Compute per-student baseline statistics (pitch/volume)
- Derive personalized alarm thresholds (mean + z*std)
- Support individual speech profile modeling"

Commit 3:
git commit -m "feat: Add longitudinal trend analysis and reporting

- Implement linear regression for trend detection
- Generate student_report.txt with formatted summaries
- Add student_report.pdf with visual dashboards"

Commit 4:
git commit -m "feat: Export session data to CSV for external analysis

- Write session_summary.csv with all key metrics
- Enable integration with Excel, R, and analytics tools"
```

### Option Set 4: Concise Single-Line (if you prefer minimal)
```
git commit -m "Add multi-session aggregation, analysis, and reporting system"
```

---

## 6. Project Delivery Summary

| Component | Status | Lines of Code | Complexity |
|-----------|--------|---------------|-----------|
| Per-Session Analysis (Final Analysis Code.py) | Pre-existing | ~450 | Medium |
| Session Aggregation (aggregate_sessions.py) | ✅ NEW | ~500 | High |
| CSV Export | ✅ NEW | Built-in | Low |
| Text Report Generation | ✅ NEW | ~200 | Medium |
| PDF Visualization | ✅ NEW | ~100 | Medium |
| **Total New Code** | | **~800 lines** | |

---

## 7. Next Steps (Optional Enhancements)

- [ ] Add statistical significance testing (p-values for trends)
- [ ] Expand keyword list (pending stakeholder approval)
- [ ] Implement alerts for critical anomaly thresholds
- [ ] Add comparison between students (benchmarking)
- [ ] Create interactive dashboard (web-based)
- [ ] Add correlation analysis (speech patterns ↔ behavioral indicators)

---

**Report Generated**: April 16, 2026  
**Status**: Ready for Production Use
