from pathlib import Path

from langchain_core.tools import tool
from openai import OpenAI

@tool("transcribe-audio")
def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file (mp3, wav, m4a) into text."""
    client = OpenAI()
    with Path(file_path).open("rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
    return transcript.text
