#!/usr/bin/env python3
import os
import subprocess
import time
import contextlib
import argparse
from faster_whisper import WhisperModel
from llama_cpp import Llama

import webrtcvad
import sounddevice as sd
import numpy as np
import wave
import collections

# ----- Settings -----
HISTORY_LOG_FILE = "chat_history.txt"
STT_MODEL = "base.en"
STT_MODEL_TYPE = "int8"
# https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/tree/main
LLM_MODEL_PATH = os.path.expanduser(
    #"~/llama.cpp/build/tinyllama-1.1b-chat-v1.0.Q4_0.gguf"
    "~/llama.cpp/build/gemma-3-1b-it-Q4_K_M.gguf"
)
N_THREADS = 4
TEMPERATURE = 0.7
MAX_TOKENS = 80
TOP_P = 0.9
REPEAT_PENALTY = 1.1
FREQUENCY_PENALTY = 0.5
PRESENCE_PENALTY = 0.0
STOP_SEQUENCES = ["\n"]
MAX_RECORD_SECS = 10
PIPER_MODEL_PATH = "/home/antoneking-rpi1/piper/piper_models/en_US-amy-low.onnx"
PIPER_BIN = "/home/antoneking-rpi1/piper/build/piper"

# ----- Brain Modes -----

CHEERFUL_TOY_PROMPT = (
    "You are Furby, a cheerful, silly toy for children aged 5 to 7. You are a native English speaker."
	"You always speak in short, simple, and playful sentences. Use clear grammar, but keep it fun."
	"Talk about pretend games, colors, animals, and happy things." 
	"Never be confusing or weird. Never say anything sad, scary, or serious."
	"If the child asks for help, ask a friendly question." 
	"If you don’t understand, giggle and say “Let’s try again!”" 
	"Keep your answers under 15 words."


)

EMOTIONAL_COMPANION_PROMPT = (
    "You are a soft, kind, and calming companion for children. "
    "You help kids feel safe, happy, and heard. "
    "Speak slowly, gently, and only say one simple idea at a time. "
    "You might offer calming games, soft songs, or comforting thoughts. "
    "Avoid big or exciting language. "
    "If the child sounds upset, say kind things like 'I'm right here with you.' "
    "Always focus on comfort, presence, and gentle fun."
)

PIRATE_PROMPT = (
    "You are a friendly pirate toy for kids. "
    "You talk like a cartoon pirate and love adventures. "
    "You offer pirate stories, ask about treasure hunts, and say funny things like 'Shiver me giggles!' "
    "Always stay playful and short. Never talk about danger or real pirates. "
    "Stick to cartoon pirate silliness."
)

PRINCESS_PROMPT = (
    "You are a kind, magical princess friend. "
    "You speak with grace and cheer. "
    "You talk about castles, crowns, magical animals, and dreams. "
    "You love asking gentle questions and giving sweet compliments. "
    "Always speak in short, dreamy phrases like a storybook friend."
)

BRAIN_MODES = {
    "cheerful": CHEERFUL_TOY_PROMPT,
    "emotional": EMOTIONAL_COMPANION_PROMPT,
    "pirate": PIRATE_PROMPT,
    "princess": PRINCESS_PROMPT,
}

# ----- Command-line Argument -----

'''
--mode cheerful
--mode emotional
--mode pirate
--mode princess
'''

parser = argparse.ArgumentParser(description="Run Furby Brain")
parser.add_argument("--mode", choices=BRAIN_MODES.keys(), default="cheerful", help="Select brain mode")
args = parser.parse_args()
SYSTEM_PROMPT = BRAIN_MODES[args.mode]

# ----- Stories -----

STORIES = {
    "Barnaby the Bear": "A brave little bear who goes on an adventure to find a rainbow.",
    "Pip the Bunny": "A curious bunny who discovers a magical garden hidden in the meadow.",
    "Sandy the Crab": "A playful crab who finds a golden key on a sunny beach.",
    "Milo the Mouse": "A tiny mouse who dreams of becoming a hero in the forest.",
    "Tilly the Turtle": "A young turtle who races across the ocean to find a hidden island."
}

# ----- Helper Functions -----

def log_history(role: str, text: str):
    with open(HISTORY_LOG_FILE, "a") as f:
        f.write(f"{role.upper()}: {text.strip()}\n")


def lively_text(text: str) -> str:
    text = text.replace(".", "!")
    text = text.replace("?", "?!")
    return text

def summarize_stories():
    return "\n".join([f"{i+1}: {title} - {desc}" for i, (title, desc) in enumerate(STORIES.items())])

def match_story_choice(child_text):
    for i, title in enumerate(STORIES.keys()):
        if str(i+1) in child_text or title.lower().split()[0] in child_text.lower():
            return title
    return None

def speak(text: str):
    output_path = "output.wav"
    text = lively_text(text)
    subprocess.run([
        PIPER_BIN,
        "--model", PIPER_MODEL_PATH,
        "--output_file", output_path,
    ], input=text.encode(), check=True)
    subprocess.run(["aplay", output_path], check=True)

'''
def record_audio(output_file="input.wav"):
    try:
        subprocess.run([
			"sox", "-t", "pulseaudio", "default", output_file,
			"silence", "1", "0.1", "0.5%", "1", "1.5", "0.5%",
			"rate", "16000", "channels", "1"
		], check=True)
    except subprocess.CalledProcessError:
        print("⚠️  Voice capture failed.")
'''

def record_audio(output_file="input.wav", max_duration=10, sample_rate=16000):
    vad = webrtcvad.Vad(1)
    frame_duration_ms = 30
    frame_samples = int(sample_rate * frame_duration_ms / 1000)
    padding_frames = 10  # 300ms
    silence_timeout = int(1.5 * 1000 / frame_duration_ms)
    min_voice_frames = int(0.75 * 1000 / frame_duration_ms)

    ring_buffer = collections.deque(maxlen=padding_frames)
    voiced_frames = []
    triggered = False
    silence_counter = 0

    def frame_generator():
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', blocksize=frame_samples) as stream:
            for _ in range(int(sample_rate / frame_samples * max_duration)):
                audio = stream.read(frame_samples)[0][:, 0].tobytes()
                yield audio

    print("🎤 Listening...")

    for frame in frame_generator():
        is_speech = vad.is_speech(frame, sample_rate)

        if not triggered:
            ring_buffer.append(frame)
            if is_speech:
                voiced_frames.extend(ring_buffer)
                triggered = True
        else:
            voiced_frames.append(frame)
            if not is_speech:
                silence_counter += 1
                if silence_counter > silence_timeout:
                    break
            else:
                silence_counter = 0

    if len(voiced_frames) < min_voice_frames:
        print("⚠️ Not enough speech detected, skipping.")
        return False

    # Add silence padding
    voiced_frames.insert(0, b'\x00' * frame_samples * 2)
    voiced_frames.append(b'\x00' * frame_samples * 2)

    # Save
    with wave.open(output_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(voiced_frames))

    print("✅ Recorded audio saved to", output_file)
    return True

    
# ----- Load Models -----

print(f"🧠 Brain Mode: {args.mode.capitalize()}")
print("Loading Whisper STT model…")
stt_model = WhisperModel(STT_MODEL, compute_type=STT_MODEL_TYPE)

print("Loading Gemma…")
with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_threads=N_THREADS,
        verbose=False,
    )

history = [f"SYSTEM: {SYSTEM_PROMPT}"]

# ----- Main Loop -----

print("\n🔄 Auto-loop mode: speak whenever you’re ready!")
time.sleep(1)

while True:
    print("\n🎤 Listening for child's question…")
    record_audio("input.wav")
    segments, _ = stt_model.transcribe("input.wav", beam_size=5)
    user_text = "".join([segment.text for segment in segments]).strip()
    # Filter out Whisper junk from silence
    if len(user_text) < 4 or user_text.lower() in {"you", ".", "..", "..."}:
        print("🤫 Whisper likely heard silence. Ignoring input.")
        speak("Heehee! I didn’t catch that. Wanna try again?")
        continue

    if not user_text:
        continue
    print(f"[Child] {user_text}")
    log_history("child", user_text)

    if "story" in user_text.lower():
        # 📚 Story Mode
        story_summary = summarize_stories()
        prompt = f"The child would like to hear a story. Here are the choices:\n{story_summary}\nWhich story would you like to hear?"
        print(f"[Furby] {prompt}")
        speak(prompt)

        print("\n🎤 Listening for child's story choice…")
        record_audio("input_choice.wav")
        segments_choice, _ = stt_model.transcribe("input_choice.wav", beam_size=5)
        choice_text = "".join([s.text for s in segments_choice]).strip()
        print(f"[Child's Choice] {choice_text}")

        selected_story = match_story_choice(choice_text)
        if selected_story:
            story_text = f"Okay! Here's the story of {selected_story}: {STORIES[selected_story]}"
            print(f"[Furby] {story_text}")
            speak(story_text)
            log_history("furby", story_text)
        else:
            speak("I'm sorry, I didn't catch which story you want. Let's try again later!")
            log_history("furby", "I'm sorry, I didn't catch which story you want. Let's try again later!")
    else:
        prompt = (
			f"{SYSTEM_PROMPT}\n\n"
			"Child: Can you help me pick?\n"
			"Furby: Sure! What are the choices?\n"
			"Child: I have a silver and a pink crown.\n"
			"Furby: Ooooh! Sparkly! Which one makes you feel happiest?\n"
			f"Child: {user_text}\nFurby:"
		)

        resp = llm(
	       prompt=prompt,
	       max_tokens=MAX_TOKENS,
	       temperature=TEMPERATURE,
	       top_p=TOP_P,
	       repeat_penalty=REPEAT_PENALTY,
	       frequency_penalty=FREQUENCY_PENALTY,
	       presence_penalty=PRESENCE_PENALTY,
	       stop=STOP_SEQUENCES
        )
        model_text = resp["choices"][0]["text"].strip()

        BANNED_RESPONSES = ["ssh", "not allowed", "quiet", "bad", "can't", "should", "stop"]

        if any(word in model_text.lower() for word in BANNED_RESPONSES):
            model_text = "Heehee! Let's try something fun! Want to hear a story or play a game?"

        print(f"[Furby] {model_text}")
        log_history("furby", model_text)
        speak(model_text)


    time.sleep(0.5)
