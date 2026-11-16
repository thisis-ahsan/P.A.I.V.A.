import numpy as np
import flet as ft
import flet_audio_recorder as far
from faster_whisper import WhisperModel

# =====================================================================
# GLOBAL INITIALIZATION (VRAM Optimized)
# =====================================================================
MODEL_SIZE = "base"
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")

# Global memory buffer to store incoming microphone chunks
audio_stream_buffer = bytearray()
is_recorder_recording = False  # Track the recording state of the recorder

# =====================================================================
# MEMORY-BASED STREAM TRANSLATION
# =====================================================================
def listen(raw_pcm_bytes: bytearray) -> str:
    """
    Consumes raw PCM16 binary chunks directly from the live microphone buffer,
    normalizes it to 32-bit floating point, and runs live inference.
    """
    if len(raw_pcm_bytes) == 0:
        return ""

    # Convert raw binary bytes into signed 16-bit integers
    audio_data = np.frombuffer(raw_pcm_bytes, dtype=np.int16)
    
    # Normalize integers to float32 between -1.0 and 1.0 (Whisper's native language)
    audio_float32 = audio_data.astype(np.float32) / 32768.0

    # Process the structural float array directly in VRAM without creating local files
    segments, _ = model.transcribe(
        audio_float32, 
        beam_size=1,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400)
    )
    
    return " ".join([segment.text for segment in segments]).strip()

# =====================================================================
# USER INTERFACE (Live Capture Application)
# =====================================================================
def main(page: ft.Page):
    page.title = "Live Streaming Voice Agent"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.window_width = 500
    page.window_height = 450

    # Reactive UI states
    status_text = ft.Text("Microphone Status: Idle", size=16, weight=ft.FontWeight.BOLD)
    live_transcript = ft.TextField(
        label="Live Speech Window", 
        multiline=True, 
        read_only=True, 
        width=420, 
        min_lines=5
    )

    # 1. Capture the live data stream directly from the system hardware
    async def handle_incoming_stream(e: far.AudioRecorderStreamEvent):
        global audio_stream_buffer
        # Append incoming PCM16 binary chunk into the global byte stack
        audio_stream_buffer.extend(e.chunk)
        
        # Continuous sliding live transcription update
        text = listen(audio_stream_buffer)
        if text:
            live_transcript.value = text
            page.update()

    # 2. Configure the Recorder for Live Pipeline Streaming
    recorder = far.AudioRecorder(on_stream=handle_incoming_stream)
    page.overlay.append(recorder)

    async def toggle_streaming(e):
        global audio_stream_buffer, is_recorder_recording
        
        if not is_recorder_recording:
            if await recorder.has_permission():
                # Clean out prior audio artifacts
                audio_stream_buffer.clear()
                live_transcript.value = ""
                
                status_text.value = "Microphone Status: LIVE STREAMING"
                status_text.color = ft.Colors.RED_ACCENT
                action_btn.text = "Stop Live Stream"
                action_btn.icon = ft.Icons.STOP
                page.update()
                
                # Start recording using PCM16BITS to unlock live `on_stream` access
                is_recorder_recording = True
                await recorder.start_recording(
                    configuration=far.AudioRecorderConfiguration(
                        encoder=far.AudioEncoder.PCM16BITS, # Mandatory for streaming
                        sample_rate=16000,                  # Whisper's standard frame timing
                        channels=1
                    )
                )
            else:
                status_text.value = "Error: Microphone permission denied."
                page.update()
        else:
            # Tear down processing state clean
            is_recorder_recording = False
            await recorder.stop_recording()
            status_text.value = "Microphone Status: Disconnected"
            status_text.color = ft.Colors.TEXT
            action_btn.text = "Start Live Stream"
            action_btn.icon = ft.Icons.PLAY_ARROW
            page.update()

    action_btn = ft.ElevatedButton(
        content="Start Live Stream",
        icon=ft.Icons.PLAY_ARROW,
        on_click=toggle_streaming,
        style=ft.ButtonStyle(shape=ft.RoundedRectangleBorder(radius=8))
    )

    page.add(
        ft.Column(
            controls=[
                status_text,
                ft.Container(height=15),
                action_btn,
                ft.Container(height=25),
                live_transcript
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )
    )

if __name__ == "__main__":
    ft.run(main)