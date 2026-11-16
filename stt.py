import queue
import sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# Configuration Constants
MODEL_SIZE = "base"         # Balance between speed and bilingual accuracy
DEVICE = "cuda"               # Change to "cuda" if you have an NVIDIA GPU
COMPUTE_TYPE = "float16"        # Use "float16" for GPU, "int8" for CPU
SAMPLE_RATE = 16000          # Whisper strictly requires 16kHz audio
CHUNK_DURATION = 4           # Audio block size in seconds to send to Whisper

# Thread-safe queue to pass audio from mic thread to processing thread
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """This function is called by sounddevice for every new audio chunk."""
    if status:
        print(status, file=sys.stderr)
    # Put the audio data copy into the queue
    audio_queue.put(indata.copy())


print(f"Loading Whisper model '{MODEL_SIZE}' on {DEVICE}...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
# Calculate how many frames make up our CHUNK_DURATION
chunk_frames = int(SAMPLE_RATE * CHUNK_DURATION)
print("\n>>> Model loaded! Press Ctrl+C to stop.\n")


def listen():
    audio_buffer = np.zeros((0, 1), dtype=np.float32)
    while not audio_queue.empty():
        audio_queue.get_nowait()

    # Start the microphone listening stream
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, dtype='float32'):
        while True:
            try:
                # Pull raw audio data from the queue
                data = audio_queue.get()
                audio_buffer = np.append(audio_buffer, data, axis=0)

                # Process once we accumulate enough audio for a chunk
                if len(audio_buffer) >= chunk_frames:
                    # Flatten the array for Whisper (1D array)
                    whisper_input = audio_buffer.flatten()
                    
                    # Transcribe the accumulated chunk
                    # beam_size=1 speeds up transcription for real-time use
                    segments, _ = model.transcribe(
                        whisper_input,
                        language="en",                      # English and Hindi support
                        beam_size=1,                         # Fast decoding
                        condition_on_previous_text=False,    # Wipes the memory after every command 🛑 (CRITICAL)
                        vad_filter=True,                     # Automatically cuts out background noise/silence
                        vad_parameters=dict(min_silence_duration_ms=500),
                        initial_prompt="The assistant's name is Annie. And Ahsan Ansari is her boss, he built her himself."
                    )
                    
                    for segment in segments:
                        # Clean up text and print instantly
                        text = segment.text.strip()
                        if text:
                            return text  # Return the transcribed text for further processing (e.g., TTS)

                    # Clear buffer to listen to the next segment
                    audio_buffer = np.zeros((0, 1), dtype=np.float32)

            except Exception as e:
                print("Error:", e)