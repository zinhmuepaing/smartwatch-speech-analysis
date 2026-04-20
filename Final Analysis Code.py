import sys
import datetime
from io import StringIO
from pathlib import Path
from multiprocessing import Pool, cpu_count, Value, Lock, freeze_support

# Use a non-interactive backend so matplotlib works in child processes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Audio / signal processing
import librosa          # Audio loading + short-time analysis
import pysrt            # Subtitle (.srt) parsing
import numpy as np      # Numerical operations
import soundfile as sf  # Writing WAV files
import parselmouth      # Praat-based pitch extraction


# =============================================================================
# USER SETTINGS / ANALYSIS PARAMETERS
# =============================================================================

# Short-time analysis parameters.
# These control the time resolution of RMS and pitch measurements.
FRAME_LENGTH = 2048
HOP_LENGTH = 512

# Length (seconds) of the sliding window used for anomaly detection.
# Feature values are averaged over this window to reduce frame-level noise.
ANOMALY_WINDOW_SECONDS = 1.0

# Z-score thresholds for anomaly detection.
# At Z = 1.5 (one-tailed), ~6.68% of normally-distributed values are expected
# to exceed this threshold by chance — i.e., the threshold sits at the 93rd
# percentile. For comparison: Z = 2.0 → ~2.28%; Z = 1.96 → ~2.5%.
# Using 1.5 makes detection more sensitive, appropriate for mild speech anomalies.
PITCH_Z_THRESHOLD = 1.5
VOLUME_Z_THRESHOLD = 1.5

# Keywords to search for in subtitles
KEYWORDS_TO_FIND = ['唔該', '多謝', '早晨']

# Long recordings are split into chunks to limit memory use
# and allow parallel processing.
CHUNK_LENGTH_SECONDS = 15 * 60

# Acceptable pitch range for speech (Hz)
PITCH_FLOOR = 75
PITCH_CEILING = 500

# Plot formatting
FIGURE_WIDTH = 14
FIGURE_HEIGHT = 10
GRAPH_DPI = 150


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_srt_exists(audio_path: Path) -> Path:
    """
    Ensure an SRT file exists next to the audio file.
    If none is found, generate one using Whisper.
    """
    existing_srts = list(audio_path.parent.glob("*.srt"))

    if existing_srts:
        print(f"✔ Found existing SRT ({existing_srts[0].name}), skipping: {audio_path.name}")
        return existing_srts[0]

    print(f"📝 No SRT found for {audio_path.name}. Generating transcript...")

    import whisper
    model = whisper.load_model("tiny", device="cpu")
    result = model.transcribe(str(audio_path), task="transcribe", verbose=False)

    srt_path = audio_path.with_suffix(".srt")

    # Write Whisper output in standard SRT format
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], start=1):
            start, end = seg["start"], seg["end"]
            text = seg["text"].strip()

            def ts(t):
                h = int(t // 3600)
                m = int((t % 3600) // 60)
                s = int(t % 60)
                ms = int((t - int(t)) * 1000)
                return f"{h:02}:{m:02}:{s:02},{ms:03}"

            f.write(f"{i}\n{ts(start)} --> {ts(end)}\n{text}\n\n")

    return srt_path


def seconds_to_hms(seconds):
    """Convert seconds to HH:MM:SS string."""
    return str(datetime.timedelta(seconds=float(seconds)))


def z_score(x, mean, std):
    """
    Compute a z-score safely.
    Z = (x - mean) / std
    """
    return (x - mean) / std if std > 0 else 0.0


def format_anomalies(anomalies):
    """Pretty-print anomaly lists for text output."""
    if not anomalies:
        return "[]"
    return "[\n" + ",\n".join(
        f"    ('{seconds_to_hms(t)}', {v:.6f}, {z:.6f})"
        for t, v, z in anomalies
    ) + "\n]"


# =============================================================================
# AUDIO CHUNKING
# =============================================================================

def split_file(audio_path):
    """
    Split an audio file into fixed-length chunks.

    Why:
    - avoids loading very long files into memory
    - enables multiprocessing
    - keeps statistics locally meaningful
    """
    if "_chunk_" in audio_path.stem:
        raise RuntimeError("split_file() called on chunk file")

    y, sr = librosa.load(str(audio_path), sr=None)
    total_seconds = len(y) / sr
    chunks = []

    for i in range(int(np.ceil(total_seconds / CHUNK_LENGTH_SECONDS))):
        start = int(i * CHUNK_LENGTH_SECONDS * sr)
        end = int(min((i + 1) * CHUNK_LENGTH_SECONDS * sr, len(y)))

        out_path = audio_path.with_name(
            f"{audio_path.stem}_chunk_{i+1}.wav"
        ).resolve()

        sf.write(out_path, y[start:end], sr)
        chunks.append((out_path, i, i * CHUNK_LENGTH_SECONDS))

    return chunks


# =============================================================================
# ANALYSIS FUNCTION (WORKER PROCESS)
# =============================================================================

def analyze_audio_chunk(args):
    """
    Perform signal processing and statistical anomaly detection
    on a single audio chunk.
    """
    audio_path, chunk_index, chunk_start_sec, transcript_path = args

    # Capture stdout so logs from each worker can be collected
    buffer = StringIO()
    original_stdout = sys.stdout
    sys.stdout = buffer

    fig_data = None

    try:
        print(f"Processing {audio_path.name} (chunk {chunk_index+1})")

        # ------------------------------------------------------------------
        # LOAD AUDIO AND BASIC FEATURES
        # ------------------------------------------------------------------

        y, sr = librosa.load(audio_path, sr=None)

        # RMS energy: short-time average signal magnitude.
        # Used here as a proxy for loudness.
        rms = librosa.feature.rms(
            y=y,
            frame_length=FRAME_LENGTH,
            hop_length=HOP_LENGTH
        )[0]

        # Convert frame indices to absolute time (seconds),
        # then offset by the chunk start time.
        times = (
            librosa.frames_to_time(
                np.arange(len(rms)),
                sr=sr,
                hop_length=HOP_LENGTH
            ) + chunk_start_sec
        )

        # ------------------------------------------------------------------
        # PITCH EXTRACTION
        # ------------------------------------------------------------------

        # Praat-based pitch tracker.
        # Returns 0 Hz for unvoiced frames.
        snd = parselmouth.Sound(y, sr)
        pitch_obj = snd.to_pitch(
            time_step=HOP_LENGTH / sr,
            pitch_floor=PITCH_FLOOR,
            pitch_ceiling=PITCH_CEILING
        )

        f0 = pitch_obj.selected_array['frequency']
        f0[f0 == 0] = np.nan  # Treat unvoiced frames as missing data

        # Align feature arrays
        min_len = min(len(f0), len(rms))
        f0, rms, times = f0[:min_len], rms[:min_len], times[:min_len]

        # Only voiced frames contribute to pitch statistics
        voiced_pitch = f0[~np.isnan(f0)]

        # ------------------------------------------------------------------
        # GLOBAL STATISTICS (REFERENCE DISTRIBUTIONS)
        # ------------------------------------------------------------------

        # These describe what is "normal" for this chunk
        pitch_mean = np.nanmean(voiced_pitch) if len(voiced_pitch) else 0.0
        pitch_std = np.nanstd(voiced_pitch) if len(voiced_pitch) else 1e-6

        rms_mean = np.mean(rms)
        rms_std = np.std(rms) + 1e-6

        # ------------------------------------------------------------------
        # WINDOWED Z-SCORE ANOMALY DETECTION
        # ------------------------------------------------------------------

        # Convert window length from seconds to frames
        window_frames = max(1, int(ANOMALY_WINDOW_SECONDS * sr / HOP_LENGTH))

        abnormal_pitch = []
        abnormal_rms = []

        i = 0
        while i + window_frames <= min_len:
            p_win = f0[i:i + window_frames]
            r_win = rms[i:i + window_frames]

            # Pitch: average only voiced frames
            if np.any(~np.isnan(p_win)):
                p_avg = np.nanmean(p_win)
                z_p = z_score(p_avg, pitch_mean, pitch_std)
            else:
                z_p = np.nan

            # RMS: always defined
            r_avg = np.mean(r_win)
            z_r = z_score(r_avg, rms_mean, rms_std)

            # Flag statistically extreme deviations
            if z_p > PITCH_Z_THRESHOLD:
                abnormal_pitch.append((times[i], p_avg, z_p))

            if z_r > VOLUME_Z_THRESHOLD:
                abnormal_rms.append((times[i], r_avg, z_r))

            # Skip ahead by one full window if an anomaly is found
            # to avoid repeatedly flagging the same event.
            i += window_frames if (z_p > PITCH_Z_THRESHOLD or z_r > VOLUME_Z_THRESHOLD) else 1

        # ------------------------------------------------------------------
        # SUBTITLE KEYWORD SEARCH
        # ------------------------------------------------------------------

        subs = pysrt.open(transcript_path)
        polite_count = 0

        for sub in subs:
            for kw in KEYWORDS_TO_FIND:
                if kw in sub.text:
                    polite_count += 1
                    print(f"Found '{kw}' at {sub.start.to_time()} → {sub.end.to_time()}")

        if polite_count == 0:
            print("Polite words detected: <NONE>")

        # ------------------------------------------------------------------
        # SUMMARY OUTPUT
        # ------------------------------------------------------------------

        print("abnormal_pitch =", format_anomalies(abnormal_pitch))
        print("abnormal_rms =", format_anomalies(abnormal_rms))
        print(f"Pitch mean/std: {pitch_mean:.2f}/{pitch_std:.2f}")
        print(f"RMS mean/std: {rms_mean:.6f}/{rms_std:.6f}")
        print(f"Polite word count: {polite_count}")

        # ------------------------------------------------------------------
        # VISUALIZATION
        # ------------------------------------------------------------------

        fig, (ax1, ax2) = plt.subplots(
            2, 1,
            figsize=(FIGURE_WIDTH, FIGURE_HEIGHT),
            sharex=True
        )

        ax1.plot(times, f0, label='Pitch (Hz)')
        ax1.axhline(pitch_mean, linestyle='--', color='green')
        ax1.axhline(pitch_mean + PITCH_Z_THRESHOLD * pitch_std,
                    linestyle='--', color='red')

        for t, _, _ in abnormal_pitch:
            ax1.axvspan(t, t + ANOMALY_WINDOW_SECONDS, color='red', alpha=0.15)

        ax2.plot(times, rms, label='RMS', color='coral')
        ax2.axhline(rms_mean, linestyle='--', color='green')
        ax2.axhline(rms_mean + VOLUME_Z_THRESHOLD * rms_std,
                    linestyle='--', color='red')

        for t, _, _ in abnormal_rms:
            ax2.axvspan(t, t + ANOMALY_WINDOW_SECONDS, color='red', alpha=0.15)

        ax2.set_xlabel("Time (seconds)")

        fig_data = fig

    except Exception as e:
        print(f"ERROR processing {audio_path.name}: {e}")

    finally:
        sys.stdout = original_stdout

    # Update shared progress counter safely
    with progress_lock:
        completed_chunks.value += 1
        print(f"Progress: {completed_chunks.value} chunks processed")

    return audio_path.name, buffer.getvalue(), fig_data


# =============================================================================
# PROCESS A SINGLE FILE (CONTROLLER LOGIC)
# =============================================================================

def process_file(audio_path, transcript_path):
    """
    High-level orchestration for one audio file.

    Responsibilities:
    - split audio into chunks
    - dispatch chunk analysis in parallel
    - collect and order results
    - write text and PDF outputs
    - clean up temporary files
    """
    # Split audio into chunk files on disk
    chunks = split_file(audio_path)

    # Build task arguments for each chunk
    tasks = [
        (chunk_path, idx, start_sec, transcript_path)
        for chunk_path, idx, start_sec in chunks
    ]

    # Use half the available CPUs to avoid oversubscription
    pool_size = max(1, cpu_count() // 2)

    # Run chunk analysis in parallel
    with Pool(pool_size) as pool:
        results = pool.map(analyze_audio_chunk, tasks)

    # Sort results so chunks appear in correct order
    results_sorted = sorted(
        results,
        key=lambda x: int(x[0].split('_chunk_')[-1].split('.')[0])
    )

    # ------------------------------------------------------------------
    # WRITE TEXT OUTPUT
    # ------------------------------------------------------------------

    output_file = audio_path.with_name(f"{audio_path.stem}_analysis.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for name, text, _ in results_sorted:
            f.write(f"\n=== {name} ===\n")
            f.write(text)

    print(f"✔ Text results saved to {output_file}")

    # ------------------------------------------------------------------
    # WRITE PDF WITH ALL PLOTS
    # ------------------------------------------------------------------

    pdf_file = audio_path.with_name(f"{audio_path.stem}_analysis.pdf")
    with PdfPages(pdf_file) as pdf:
        for _, _, fig in results_sorted:
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)

    print(f"✔ PDF saved to {pdf_file}")

    # ------------------------------------------------------------------
    # CLEAN UP TEMPORARY CHUNK FILES
    # ------------------------------------------------------------------

    for chunk_path, _, _ in chunks:
        chunk_path.unlink(missing_ok=True)


# =============================================================================
# FILE DISCOVERY + MULTIPROCESSING STATE
# =============================================================================

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

wav_files = [p for p in BASE_DIR.rglob("*.wav") if "_chunk_" not in p.stem]

FILES_TO_PROCESS = []

for wav_path in wav_files:
    srt_matches = list(wav_path.parent.glob("*.srt"))
    if srt_matches:
        FILES_TO_PROCESS.append((wav_path, srt_matches[0]))
    else:
        FILES_TO_PROCESS.append((wav_path, ensure_srt_exists(wav_path)))

completed_chunks = Value("i", 0)
progress_lock = Lock()


# =============================================================================
# MAIN
# =============================================================================

def main():
    freeze_support()
    print(f"Starting analysis of {len(FILES_TO_PROCESS)} files...\n")

    for audio_path, transcript_path in FILES_TO_PROCESS:
        process_file(audio_path, transcript_path)

    print("\nAll files processed.")

if __name__ == "__main__":
    main()
