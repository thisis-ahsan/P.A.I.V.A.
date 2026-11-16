import pyttsx3
import queue
import sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import ollama
import datetime
import AppOpener as ao
import time
import threading
import keyboard


class PAIVA:
    """**P.A.I.V.A. stands for Personal AI Virtual Assistant.**
    
    Useful Functions:
        - **listen()**: Listens to the microphone and returns transcribed text.
        - **speak(text)**: Converts text to speech and plays it.
        - **think(user_input)**: Sends user input to the LLM and returns the response.
        - **main()**: Main loop that listens, thinks, and speaks.
        - **detect_wake_word()**: Listens for the wake word "Annie" and starts the assistant.
    """
    def __init__(self):
        self.name = "Annie"
        self.version = "1.0"
        self.developer = "Ahsan Ansari"


        # Speech Engine Initialization
        engine = pyttsx3.init()
        self.speech_engine = engine
        engine.setProperty("rate", 195)
        engine.setProperty("voice", engine.getProperty("voices")[1].id)


        # Speech Recognition Model Initialization
        # Configuration Constants
        self.STT_MODEL_SIZE = "base"  # Balance between speed and bilingual accuracy
        self.DEVICE = "cuda"  # Change to "cuda" if you have an NVIDIA GPU
        self.COMPUTE_TYPE = "float16"  # Use "float16" for GPU, "int8" for CPU
        self.SAMPLE_RATE = 16000  # Whisper strictly requires 16kHz audio
        self.CHUNK_DURATION = 4  # Audio block size in seconds to send to Whisper

        # Thread-safe queue to pass audio from mic thread to processing thread
        self.audio_queue = queue.Queue()

        print(f"Loading Whisper model '{self.STT_MODEL_SIZE}' on {self.DEVICE}...")
        self.stt_model = WhisperModel(self.STT_MODEL_SIZE, device=self.DEVICE, compute_type=self.COMPUTE_TYPE)
        # Calculate how many frames make up our CHUNK_DURATION
        self.chunk_frames = int(self.SAMPLE_RATE * self.CHUNK_DURATION)
        print("\n>>> Whisper Model loaded!\n")


        # Ollama Model Initialization
        SYSTEM_PROMPT = "You're Annie, a helpful AI assistant developed by and for Ahsan (pronunced as eh-sun) Ansari. Ahsan is an Indian Teen Entrepreneur, Full Stack Developer, Python Developer, Marketing and AI Expert. Your job is to help Ahsan with every possible thing. You should sound like a real human as much as possible"

        self.LLM_MODEL = "qwen3.5:2b"

        self.chat = [{
                    "role": "system",
                    "content": SYSTEM_PROMPT
                    }]

        print("Initializing Ollama model...")

        ollama.chat(
            model=self.LLM_MODEL,
            messages=self.chat,
            keep_alive=True
        )

        print("\n>>> Ollama Model initialized!\n")

        # Tool Definitions
        self.tools = [self.openapp, self.closeapp, self.gettime, self.speak, self.stop]
        self.available_tools = {
            "openapp": self.openapp,
            "closeapp": self.closeapp,
            "gettime": self.gettime,
            "speak": self.speak,
            "stop": self.stop
        }

        # Assistant's Processes
        self.main_process = threading.Thread(target=self.main)

        self.stop_event = threading.Event()  # Event to signal stopping the assistant
        self.speech_lock = threading.Lock()  # Lock to ensure only one speech at a time

        # Show that the assistant is ready
        print(f"\n>>> {self.name} v{self.version} by {self.developer} is ready! Say 'Annie' to wake me up.\n")


    def audio_callback(self, indata, frames, time, status):
        """This function is called by sounddevice for every new audio chunk."""
        if status:
            print(status, file=sys.stderr)
        # Put the audio data copy into the queue
        self.audio_queue.put(indata.copy())

    def listen(self):
        """Listen to the microphone and return transcribed text."""
        audio_buffer = np.zeros((0, 1), dtype=np.float32)
        while not self.audio_queue.empty():
            self.audio_queue.get_nowait()
        
            # Start the microphone listening stream
        with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1, callback=self.audio_callback, dtype='float32'):
            while True:
                try:
                    # Pull raw audio data from the queue
                    data = self.audio_queue.get()
                    audio_buffer = np.append(audio_buffer, data, axis=0)
        
                    # Process once we accumulate enough audio for a chunk
                    if len(audio_buffer) >= self.chunk_frames:
                        # Flatten the array for Whisper (1D array)
                        whisper_input = audio_buffer.flatten()
                            
                        # Transcribe the accumulated chunk
                        # beam_size=1 speeds up transcription for real-time use
                        segments, _ = self.stt_model.transcribe(
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
                                return text  # Return the transcribed text for further processing
        
                        # Clear buffer to listen to the next segment
                        audio_buffer = np.zeros((0, 1), dtype=np.float32)
        
                except Exception as e:
                    print("Error:", e)

    def speak(self, text):
        """Speak in a worker thread; press TAB to interrupt."""

        def speech_worker():
            with self.speech_lock:
                self.speech_engine.say(text)
                self.speech_engine.runAndWait()

        speech_thread = threading.Thread(
            target=speech_worker,
            daemon=True
        )
        speech_thread.start()

        try:
            while speech_thread.is_alive():
                if keyboard.is_pressed("tab"):
                    print("Skipping speech...")
                    self.speech_engine.stop()
                    break

                time.sleep(0.01)
        finally:
            speech_thread.join()

    def think(self, user_input):
        """Send user input to the LLM and return the response."""
        self.chat.append({
            "role": "user",
            "content": user_input,
        })

        while True:
            response = ollama.chat(
                model="qwen3.5:2b",
                messages=self.chat,
                tools=self.tools,
            )

            # Preserve the assistant response and its tool calls.
            self.chat.append(response.message)

            # If there are no tool calls, this is the final answer.
            if not response.message.tool_calls:
                answer = response.message.content
                print("Annie:", answer)
                return answer

            # Execute all tool calls before asking the model again.
            for tc in response.message.tool_calls:
                tool_name = tc.function.name
                arguments = tc.function.arguments

                print(f"Executing tool: {tool_name}")
                print(f"Arguments: {arguments}")

                if tool_name not in self.available_tools:
                    result = f"Error: Unknown tool '{tool_name}'."
                else:
                    try:
                        result = self.available_tools[tool_name](**arguments)
                    except Exception as e:
                        result = f"Tool execution failed: {e}"

                print(f"Tool result: {result}")

                self.chat.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "content": str(result),
                })

    def main(self):
        """Main loop that listens, thinks, and speaks."""
        while True:
            print("Listening...")
            user_input = self.listen()

            print("You:", user_input)

            user_input = user_input.strip()
            print("Thinking...")
            response = self.think(user_input)
            print("Speaking...")
            self.speak(response)

    def detect_wake_word(self):
        """Detect if the wake word 'Annie' is present in the text."""
        while True:
            mic_input = self.listen()
            print(mic_input)

            if mic_input and "annie" in mic_input.lower():
                print("Wake word detected! Starting assistant...")
                greeting = self.think("Greet me in short like a very intelligent AI assistant like J.A.R.V.I.S. from Iron Man, but also very funny and sarcastic. Make it sound like a real human.")
                self.speak(greeting)
                print("Starting main loop...")
                self.main_process.start()  # Start the main loop process after greeting

    def stop(self):
        """Stop the assistant"""
        print("Going to Sleep...")
        self.stop_event.set()  # Signal the main loop to stop

    def gettime(self) -> str:
        """Return the current local time."""
        print("Fetching latest date and time...")
        return datetime.datetime.now().strftime("%H:%M:%S")


    def openapp(self, app_name: str) -> str:
        """Open an application by its name."""
        try:
            ao.open(app_name)
            return f"Application '{app_name}' opened successfully."
        except Exception as e:
            return f"Failed to open {app_name}: {e}"


    def closeapp(self, app_name: str) -> str:
        """Close an application by its name."""
        try:
            ao.close(app_name)
            return f"Application '{app_name}' closed successfully."
        except Exception as e:
            return f"Failed to close {app_name}: {e}"


if __name__ == "__main__":
    paiva = PAIVA()
    # paiva.__init__()
    paiva.detect_wake_word()  # Start the assistant and listen for the wake word

    if keyboard.is_pressed("ctrl+c"):
        print("Exiting...")
        paiva.stop()  # Stop the assistant if "esc" is pressed
