# Smartwatch Speech Analytics for Autistic Students

## Project overview
This project analyzes smartwatch speech data collected from students with autism spectrum disorder. The watch records speech transcriptions, pitch measurements, and volume measurements from a Bluetooth lapel microphone. The main goal is to evaluate polite speech usage and identify when volume or pitch thresholds trigger alerts.

## Deliverables
1. Count the number of target empathetic keywords in each student transcript.
2. Count how many times each student exceeded the volume and pitch thresholds that triggered alarms.
3. Analyze whether students improved over time by looking at session trends.
4. Calculate the mean and standard deviation for pitch and volume for each student.
5. Produce a descriptive report summarizing findings and recommendations.

## Codebase review
The current workspace contains two main scripts:

### Final Analysis Code.py

- Main analysis engine.
-Recursively discovers .wav files.
- Ensures .srt transcripts exist, generating them with Whisper if needed.
- Splits long audio into 15-minute chunks.
- Extracts:
1. pitch via parselmouth
2. loudness (RMS) via librosa
- Detects anomalies using chunk-level z-scores.
- Counts keyword hits from .srt subtitles.
- Writes per-file <name>_analysis.txt and <name>_analysis.pdf.

### fix_srt_timestamps.py

- Auxiliary script for broken .srt timestamps.
- Not part of the main analytics pipeline, only needed for corrupted files.

## What the existing script already covers
- It finds `.wav` audio files and processes them automatically.
- It ensures `.srt` transcripts are available, generating them with Whisper if needed.
- It computes pitch using `parselmouth` and volume proxy using `librosa` RMS.
- It flags anomalous pitch and volume events using z-score thresholds.
- It counts target keywords from subtitle files.
- It generates per-session text reports and diagnostic PDFs.

## What the existing script does not fully provide yet
- It does not aggregate results across multiple sessions for the same student.
- It does not create a structured summary table for student-level analytics.
- It does not perform longitudinal trend analysis over time.
- It does not explicitly extract student IDs and session dates for grouping.
- It does not generate a final descriptive report with personalized threshold recommendations.

## Current Plans Considered
1. Keep the current audio and subtitle processing pipeline for pitch, volume, and keyword detection.
2. Add a structured data layer using `pandas` to save one row per session.
3. Use that session table to compare sessions, group by student, and analyze trends.
4. Calculate per-student mean and standard deviation for pitch and volume.
5. Recommend personalized alarm thresholds based on each student�s own baseline.
6. Use plots and summary metrics to show improvement or changes over time.

## Planned next steps
- I will add more polite keyword variants beyond `['??', '??', '??']` because the current list is limited.
- I will create a summary table with columns such as `student_id`, `session_id`, `session_date`, `keyword_count`, `pitch_alarm_count`, `volume_alarm_count`, `pitch_mean`, `pitch_std`, `volume_mean`, `volume_std`, and `duration_sec`.
- I will perform per-student aggregation so we can compare different sessions for the same student.
- I will analyze improvement by sorting sessions by date and checking whether undesirable speech events decrease over time.
- I will calculate per-student baselines so future alarm thresholds can be personalized rather than fixed.

## Keywords to consider adding
- more polite phrases in Cantonese and Chinese such as thank-you expressions, appreciation phrases, and polite request phrases.
- if the boss approves, I will expand the keyword list to capture more empathetic speech.


