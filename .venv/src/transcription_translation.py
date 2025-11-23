from faster_whisper import WhisperModel, BatchedInferencePipeline
import re
import subprocess
import os
from dotenv import load_dotenv
load_dotenv()


def format_timestamp_srt(seconds: float) -> str:
    """Format timestamp for SRT subtitle format"""
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    total_sec = total_ms // 1000

    hh = total_sec // 3600
    mm = (total_sec % 3600) // 60
    ss = total_sec % 60

    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def split_into_sentences(text):
    """Split text into sentences"""
    sentences = re.split(r'([.!?]+\s+)', text)
    result = []
    for i in range(0, len(sentences)-1, 2):
        sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
        if sentence.strip():
            result.append(sentence.strip())
    if len(sentences) % 2 == 1 and sentences[-1].strip():
        result.append(sentences[-1].strip())
    return result


def split_by_length(text, max_chars=42):
    """Split text into lines with max character limit"""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1
        if current_length + word_length > max_chars and current_line:
            lines.append(' '.join(current_line))
            current_line = [word]
            current_length = len(word)
        else:
            current_line.append(word)
            current_length += word_length
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines


def format_subtitle_text(text, max_chars=42, max_lines=2):
    """Format subtitle text like professional movies"""
    sentences = split_into_sentences(text)
    
    formatted_chunks = []
    for sentence in sentences:
        lines = split_by_length(sentence, max_chars)
        
        for i in range(0, len(lines), max_lines):
            chunk_lines = lines[i:i+max_lines]
            formatted_chunks.append('\n'.join(chunk_lines))
    
    return formatted_chunks


def split_segment_smart(segment, max_duration=5.0, max_chars=42):
    """Split segment intelligently with proper formatting"""
    duration = segment.end - segment.start
    text = segment.text.strip()
    
    # Always apply English formatting since we're translating to English
    text_chunks = format_subtitle_text(text, max_chars)
    
    if not text_chunks:
        return []
    
    if len(text_chunks) == 1 and duration <= max_duration:
        return [{
            'start': segment.start,
            'end': segment.end,
            'text': text_chunks[0]
        }]
    
    if hasattr(segment, 'words') and segment.words and len(segment.words) > 0:
        chunks = []
        words = list(segment.words)
        word_idx = 0
        
        for text_chunk in text_chunks:
            chunk_word_count = len(text_chunk.split())
            if word_idx >= len(words):
                break
            
            start_time = words[word_idx].start
            end_idx = min(word_idx + chunk_word_count, len(words))
            end_time = words[end_idx - 1].end
            
            if end_time - start_time > max_duration:
                end_time = start_time + max_duration
            
            chunks.append({
                'start': start_time,
                'end': end_time,
                'text': text_chunk
            })
            
            word_idx = end_idx
        
        return chunks
    
    chunks = []
    time_per_chunk = duration / len(text_chunks)
    
    for i, text_chunk in enumerate(text_chunks):
        start_time = segment.start + (i * time_per_chunk)
        end_time = min(start_time + min(time_per_chunk, max_duration), segment.end)
        
        chunks.append({
            'start': start_time,
            'end': end_time,
            'text': text_chunk
        })
    
    return chunks


def transcribe_to_srt(
    input_path,
    output_path="subtitles.srt",
    model_size="large-v3",
    device="cuda",
    compute_type="float16",
    cpu_threads=12,
    batch_size=24,
    max_subtitle_duration=8.0,
    auto_translate=True,
):
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads
    )

    batched_model = BatchedInferencePipeline(model=model)

    # First pass: detect language
    segments_detect, info = batched_model.transcribe(
        input_path,
        batch_size=batch_size,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=300,
            threshold=0.2,
            min_speech_duration_ms=100,
        ),
        language=None,
        word_timestamps=True,
    )

    detected_language = info.language
    print(f"Detected language: {detected_language}")
    
    is_english = detected_language.lower() == "en"
    needs_translation = auto_translate and not is_english
    
    if needs_translation:
        print(f"Translating {detected_language} to English...")
        segments, info = batched_model.transcribe(
            input_path,
            batch_size=batch_size,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,
                threshold=0.1,
                min_speech_duration_ms=100,
            ),
            language=detected_language,  # Specify detected language
            task="translate",  # Add translation task
            word_timestamps=True,
        )
        print("Translation complete!")
    else:
        segments = segments_detect
        if is_english:
            print("Audio is in English, no translation needed.")
        else:
            print(f"Keeping original {detected_language} text.")

    # Process and split segments
    all_chunks = []
    for seg in segments:
        # Apply formatting for English (either original or translated)
        chunks = split_segment_smart(seg, max_subtitle_duration)
        all_chunks.extend(chunks)

    # Save as SRT format with UTF-8-sig for better compatibility
    with open(output_path, "w", encoding="utf-8-sig") as f:
        for idx, chunk in enumerate(all_chunks, start=1):
            start = format_timestamp_srt(chunk['start'])
            end = format_timestamp_srt(chunk['end'])
            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{chunk['text']}\n\n")

    print(f"Subtitles saved to {output_path}")
    return output_path


def play_with_vlc(video_path, subtitle_path):
    """Play with enhanced subtitle settings"""
    vlc_path = r"C:\Program Files\VideoLAN\VLC\vlc.exe"
    
    subprocess.run([
        vlc_path,
        video_path,
        f"--sub-file={subtitle_path}",
        "--no-sub-autodetect-file",
        "--freetype-rel-fontsize=18",
        "--freetype-font=Yu Gothic",
        "--sub-text-scale=60",
        "--subsdec-encoding=UTF-8",
        "--fullscreen",
    ])


if __name__ == "__main__":
    input_video =os.getenv('in_video')
    output_srt = os.getenv('out_srt')

    transcribe_to_srt(input_video, output_srt, max_subtitle_duration=8.0, auto_translate=True)
    
    play_with_vlc(input_video, output_srt)
