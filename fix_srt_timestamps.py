import os
import shutil
from pydub import AudioSegment

def is_audio_file(file_name):
    return file_name.lower().endswith(('.wav', '.mp3', '.ogg'))

def get_audio_duration(audio_path):
    audio = AudioSegment.from_file(audio_path)
    duration = len(audio) / 1000  # Convert milliseconds to seconds
    return duration

def chunk_audio_and_transcript(audio_path, transcript_path, output_dir):
    audio = AudioSegment.from_file(audio_path)
    duration = get_audio_duration(audio_path)
    segment_length = 900  # 900 seconds = 15 minutes

    num_segments = int((duration + segment_length - 1) / segment_length)

    for i in range(num_segments):
        start_time = i * segment_length
        end_time = min(start_time + segment_length, len(audio))
        chunk_audio = audio[start_time:end_time]

        # Create output directory if it doesn't exist
        chunk_dir = os.path.join(output_dir, f'chunk_{i+1}')
        os.makedirs(chunk_dir, exist_ok=True)

        # Save chunked audio and transcript
        chunk_audio_path = os.path.join(chunk_dir, 'audio.mp3')
        chunk_transcript_path = os.path.join(chunk_dir, 'transcript.txt')

        chunk_audio.export(chunk_audio_path, format='mp3')
        shutil.copy(transcript_path, chunk_transcript_path)

def create_files_to_analyse_directory(original_root, output_root):
    for dirpath, _, filenames in os.walk(original_root):
        relative_path = os.path.relpath(dirpath, original_root)
        output_dir = os.path.join(output_root, relative_path)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in filenames:
            if is_audio_file(filename) or filename.endswith('.txt'):
                audio_path = os.path.join(dirpath, filename)
                transcript_path = audio_path.replace('audio', 'transcript').replace('.mp3', '.txt')
                
                if not os.path.exists(transcript_path):
                    continue

                chunk_dir = os.path.join(output_dir, f'chunk_1')
                os.makedirs(chunk_dir, exist_ok=True)

                # Copy original audio and transcript to the chunk directory
                shutil.copy(audio_path, os.path.join(chunk_dir, 'audio.mp3'))
                shutil.copy(transcript_path, os.path.join(chunk_dir, 'transcript.txt'))

# Main function to orchestrate the process
def main():
    original_root = '/path/to/local/original/directory'  # Update this with your local directory path
    output_root = '/path/to/local/files_to_analyse'  # Update this with your desired output directory path

    create_files_to_analyse_directory(original_root, output_root)

if __name__ == '__main__':
    main()