
import os
import tempfile
import edge_tts


class EdgeTTS():
    """
    CAREFUl! This is using microsoft remote services :)
    
    Use for online ressources OR Q/A kinda interactions.
    """
    
    def synthesize(self, speech: str) -> bytes:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        path = tmp.name
        tmp.close()

        try:
            communicate = edge_tts.Communicate(
                text=speech,
                voice="en-US-JennyNeural"
            )

            communicate.save_sync(path)

            with open(path, "rb") as f:
                audio_bytes = f.read()

            return audio_bytes

        finally:
            os.remove(path)
            
