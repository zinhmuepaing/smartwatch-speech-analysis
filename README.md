AUDIO ANALYSIS PIPELINE – README
===============================

There are 2 scripts in this parent folder: fix_srt_timestamps.py and Final Analysis Code.py

fix_srt_timestamps.py was only required for FITS_Parents Data since the original .srt files had their timestamps formatted in a way that pysrt could not read, so running it corrects it.

Final Analysis Code.py is the script that does the analysis and generates reports. Details for this script is as follows:

OVERVIEW
--------
This script performs automated analysis of speech audio files (.wav) to detect
statistically unusual behavior in:

- Pitch (fundamental frequency)
- Loudness (RMS energy)

It also scans subtitle (.srt) files for specific keywords and produces:

1. A detailed text report
2. A PDF containing diagnostic plots

The system is designed to handle long recordings efficiently using chunking
and multiprocessing.


WHAT THE SCRIPT DOES (HIGH LEVEL)
---------------------------------
For each .wav file found in the directory tree:

1. Ensures a subtitle (.srt) file exists
   - If missing, it is automatically generated using Whisper
2. Splits the audio into fixed-length chunks
3. Processes chunks in parallel:
   - Extracts pitch and loudness features
   - Computes statistical baselines
   - Detects anomalies using z-scores
   - Searches subtitles for keywords
   - Generates plots
4. Aggregates results into:
   - One text report (.txt)
   - One PDF with plots (.pdf)
5. Cleans up temporary chunk files


INPUTS
------
- One or more `.wav` audio files
- Optional `.srt` subtitle files in the same directory

If no `.srt` is found, one will be generated automatically.


OUTPUT FILES
------------
For each input audio file `<name>.wav`, the script produces:

1. `<name>_analysis.txt`
   - Per-chunk logs
   - Detected pitch and volume anomalies
   - Global statistics (mean, std)
   - Keyword detection results

2. `<name>_analysis.pdf`
   - Time-series plots of pitch and RMS
   - Threshold lines
   - Highlighted anomalous regions


KEY CONCEPTS
------------

Pitch (F0):
- Extracted using Praat via the parselmouth library
- Unvoiced frames are ignored (treated as NaN)
- Statistics are computed only on voiced frames

Loudness:
- Approximated using RMS energy
- Computed on short overlapping frames

Windowing:
- Features are averaged over fixed-length time windows
- This reduces noise and false positives

Anomaly Detection:
- Uses z-scores:
      z = (value - mean) / standard_deviation
- A window is flagged if its z-score exceeds a threshold

Thresholds:
- Pitch: 2.0 standard deviations above mean
- RMS:   1.5 standard deviations above mean

These values assume roughly normal distributions and can be adjusted.


DIRECTORY STRUCTURE
-------------------
The script recursively searches from its own directory:

.
├── analyze_audio.py
├── README.txt
├── audio1.wav
├── audio1.srt
├── audio2.wav
└── subfolder/
    └── audio3.wav


HOW TO RUN
----------
1. Install dependencies (see below)
2. Place `.wav` files in the same directory or subdirectories
3. Run:

    python analyze_audio.py

The script will automatically discover files and begin processing.


DEPENDENCIES
------------
Python 3.9+ recommended.

Required packages:
- numpy
- librosa
- soundfile
- matplotlib
- pysrt
- parselmouth
- whisper (openai-whisper)

Example installation:

    pip install numpy librosa soundfile matplotlib pysrt praat-parselmouth openai-whisper


MULTIPROCESSING NOTES
---------------------
- The script uses half of available CPU cores by default so that it can run in the background if necessary
- Matplotlib runs in a non-GUI backend for Windows compatibility
- Temporary chunk files are written to disk and deleted afterward


CUSTOMIZATION
-------------
Key parameters can be adjusted at the top of the script:

- FRAME_LENGTH / HOP_LENGTH
- ANOMALY_WINDOW_SECONDS
- PITCH_Z_THRESHOLD
- VOLUME_Z_THRESHOLD
- CHUNK_LENGTH_SECONDS
- KEYWORDS_TO_FIND

These allow tuning for different speaking styles, recording conditions,
or analysis goals.


LIMITATIONS
-----------
- Assumes relatively clean speech audio
- Z-score thresholds assume roughly unimodal distributions
- Pitch extraction may degrade with heavy noise or music


End of README.
