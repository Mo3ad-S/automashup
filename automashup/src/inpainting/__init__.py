# Audio inpainting modules

from automashup.src.inpainting.audio_inpainter import AudioInpainter
from automashup.src.inpainting.silence_handler import SilenceHandler

# Optional: Stable Audio inpainter (requires stable-audio-tools)
try:
    from automashup.src.inpainting.stable_audio_inpainter import StableAudioInpainter
except ImportError:
    StableAudioInpainter = None

__all__ = ['AudioInpainter', 'SilenceHandler', 'StableAudioInpainter']
