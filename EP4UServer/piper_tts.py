import io
import wave
from piper import PiperVoice

class PiperTTS():
    """
    Piper TTS (fully in-memory)
    Contract: returns MP3 bytes
    """

    def __init__(self, model_path: str, use_cuda: bool = False):
        self.voice = PiperVoice.load(model_path, use_cuda=use_cuda)
        print('Finished loading')

    def synthesize(self, text: str) -> bytes:
        # 1. In-memory WAV buffer
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file)

        # IMPORTANT: reset pointer
        wav_buffer.seek(0)

        return wav_buffer.getvalue()
