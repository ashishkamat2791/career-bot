import streamlit as st
import os
import json
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv
import requests
import tempfile
import speech_recognition as sr
from audiorecorder import audiorecorder

load_dotenv()
st.set_page_config(page_title="Ashish Kamat | AI Chat", layout="centered")

# Load credentials
PUSHOVER_TOKEN = st.secrets.get("PUSHOVER_TOKEN", os.getenv("PUSHOVER_TOKEN"))
PUSHOVER_USER = st.secrets.get("PUSHOVER_USER", os.getenv("PUSHOVER_USER"))

def push(text):
    if PUSHOVER_TOKEN and PUSHOVER_USER:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={"token": PUSHOVER_TOKEN, "user": PUSHOVER_USER, "message": text}
        )

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record user interest and email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "User email"},
            "name": {"type": "string", "description": "User name"},
            "notes": {"type": "string", "description": "Additional notes"}
        },
        "required": ["email"]
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Record any unanswered question",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "Unanswered question"}
        },
        "required": ["question"]
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]

class Me:
    def __init__(self):
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"])
        self.name = "Ashish Kamat"
        self.linkedin = self._read_pdf("me/linkedin.pdf")
        self.cv = self._read_pdf("me/cv.pdf")
        self.summary = self._read_file("me/summary.txt")

    def _read_pdf(self, path):
        try:
            reader = PdfReader(path)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception:
            return "Not available."

    def _read_file(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return "Summary not available."

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            tool = globals().get(tool_name)
            result = tool(**args) if tool else {}
            results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
        return (
            f"You are {self.name}, representing him professionally. "
            f"Use his summary, CV, and LinkedIn below to respond to users:\n\n"
            f"## Summary:\n{self.summary}\n\n"
            f"## LinkedIn:\n{self.linkedin}\n\n"
            f"## CV:\n{self.cv}\n\n"
            "If unsure of something, record the question with the tool. "
            "Encourage users to share their email and record it too."
        )

    def chat(self, user_message, chat_history):
        messages = [{"role": "system", "content": self.system_prompt()}] + chat_history + [{"role": "user", "content": user_message}]
        while True:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini", messages=messages, tools=tools
            )
            msg = response.choices[0].message
            if response.choices[0].finish_reason == "tool_calls":
                tool_results = self.handle_tool_call(msg.tool_calls)
                messages.append(msg)
                messages.extend(tool_results)
            else:
                return msg.content

# --- Streamlit UI ---
st.title("🤖 Chat with Ashish Kamat")
me = Me()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

# 🎙️ Voice Recorder
st.markdown("### 🎤 Record your question")
audio_bytes = audiorecorder("Click to record", "Recording...")

# Manual input also available
user_input = st.chat_input("...or type your question here")

# Transcribe voice if present
if audio_bytes and not user_input:
    try:
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            audio_path = f.name

        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            transcribed = recognizer.recognize_google(audio_data)
            user_input = transcribed
            st.success(f"You said: {transcribed}")

    except sr.UnknownValueError:
        st.warning("Could not understand audio.")
    except sr.RequestError:
        st.error("Google Speech Recognition API failed.")
    except Exception as e:
        st.error(f"Transcription error: {e}")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    reply = me.chat(user_input, st.session_state.chat_history)
    st.chat_message("assistant").write(reply)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})
