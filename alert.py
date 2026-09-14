import io, os, wave
import numpy as np


class AlertManager:
    """Manages continuous audio alarm generation and state-based playback control."""

    def __init__(self, sound_path: str = "assets/alarm.wav"):
        self.is_playing, self.total_episodes = False, 0
        self.audio_bytes = self._init_audio(sound_path)

    def _init_audio(self, path: str) -> bytes:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return f.read()
        sr, dur = 44100, 1.0
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        # Pulsing two-tone alert siren (alternating 950Hz and 1200Hz)
        freq = 950 + 250 * np.sin(2 * np.pi * 4 * t)
        phase = 2 * np.pi * np.cumsum(freq) / sr
        tone = (np.sin(phase) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(tone.tobytes())
        return buf.getvalue()

    def update(self, is_drowsy: bool) -> str:
        """
        AUDIO PLAYBACK LOGIC:
        - When entering Drowsy (awake -> drowsy): returns 'start' to begin continuous looping alarm.
        - While remaining Drowsy: returns 'keep' so the looping sound plays uninterrupted.
        - When returning to Alert (drowsy -> awake): returns 'stop' to silence the alarm immediately.
        """
        if is_drowsy:
            if not self.is_playing:
                self.is_playing = True
                self.total_episodes += 1
                return "start"
            return "keep"
        else:
            if self.is_playing:
                self.is_playing = False
                return "stop"
            return "keep"
