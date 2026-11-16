import time
import pyttsx3
from multiprocessing import Process
import keyboard


engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty("rate", 190)
engine.setProperty("voice", voices[1].id)

def speech_worker(text):
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def speak(text):
    p = Process(target=speech_worker, args=(text,))
    p.start()

    while p.is_alive():
        if keyboard.is_pressed("tab"):
            print("Skipping speech...")
            p.terminate()
            break
        time.sleep(0.01)  # Sleep briefly to avoid busy waiting

    p.join()  # Ensure the process has finished before returning    
