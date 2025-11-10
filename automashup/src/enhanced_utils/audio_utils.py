"""
Enhanced audio utilities.
"""
import numpy as np
import librosa
from typing import Tuple, Optional


def ensure_same_length(audio1: np.ndarray, audio2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensure two audio arrays have the same length.
    
    Args:
        audio1: First audio array
        audio2: Second audio array
    
    Returns:
        Tuple of (audio1, audio2) with same length
    """
    max_length = max(len(audio1), len(audio2))
    
    if len(audio1) < max_length:
        audio1 = np.pad(audio1, (0, max_length - len(audio1)), mode='constant')
    else:
        audio1 = audio1[:max_length]
    
    if len(audio2) < max_length:
        audio2 = np.pad(audio2, (0, max_length - len(audio2)), mode='constant')
    else:
        audio2 = audio2[:max_length]
    
    return audio1, audio2


def normalize_audio(audio: np.ndarray, target_max: float = 0.95) -> np.ndarray:
    """
    Normalize audio to target maximum.
    
    Args:
        audio: Input audio
        target_max: Target maximum value
    
    Returns:
        Normalized audio
    """
    if len(audio) == 0:
        return audio
    
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        normalized = audio * (target_max / max_val)
    else:
        normalized = audio
    
    return normalized


def detect_clipping(audio: np.ndarray, threshold: float = 0.99) -> bool:
    """
    Detect clipping in audio.
    
    Args:
        audio: Input audio
        threshold: Clipping threshold
    
    Returns:
        True if clipping detected
    """
    if len(audio) == 0:
        return False
    
    return np.any(np.abs(audio) > threshold)


def calculate_rms(audio: np.ndarray) -> float:
    """
    Calculate RMS energy.
    
    Args:
        audio: Input audio
    
    Returns:
        RMS energy
    """
    if len(audio) == 0:
        return 0.0
    
    return np.sqrt(np.mean(audio**2))


def calculate_snr(audio: np.ndarray) -> float:
    """
    Calculate signal-to-noise ratio.
    
    Args:
        audio: Input audio
    
    Returns:
        SNR in dB
    """
    if len(audio) == 0:
        return 0.0
    
    signal_power = np.mean(audio**2)
    noise_floor = np.percentile(np.abs(audio), 10)**2
    
    if noise_floor > 0:
        snr_db = 10 * np.log10(signal_power / noise_floor)
        return max(0.0, snr_db)
    
    return 0.0


def apply_fade_in(audio: np.ndarray, fade_duration: float, sr: int = 44100) -> np.ndarray:
    """
    Apply fade-in to audio.
    
    Args:
        audio: Input audio
        fade_duration: Fade duration in seconds
        sr: Sample rate
    
    Returns:
        Audio with fade-in
    """
    if len(audio) == 0:
        return audio
    
    fade_samples = int(fade_duration * sr)
    fade_samples = min(fade_samples, len(audio))
    
    fade_curve = np.linspace(0.0, 1.0, fade_samples)
    
    faded = audio.copy()
    faded[:fade_samples] *= fade_curve
    
    return faded


def apply_fade_out(audio: np.ndarray, fade_duration: float, sr: int = 44100) -> np.ndarray:
    """
    Apply fade-out to audio.
    
    Args:
        audio: Input audio
        fade_duration: Fade duration in seconds
        sr: Sample rate
    
    Returns:
        Audio with fade-out
    """
    if len(audio) == 0:
        return audio
    
    fade_samples = int(fade_duration * sr)
    fade_samples = min(fade_samples, len(audio))
    
    fade_curve = np.linspace(1.0, 0.0, fade_samples)
    
    faded = audio.copy()
    faded[-fade_samples:] *= fade_curve
    
    return faded


def apply_crossfade(audio1: np.ndarray, audio2: np.ndarray, 
                   fade_duration: float, sr: int = 44100) -> np.ndarray:
    """
    Apply crossfade between two audio segments.
    
    Args:
        audio1: First audio segment
        audio2: Second audio segment
        fade_duration: Fade duration in seconds
        sr: Sample rate
    
    Returns:
        Crossfaded audio
    """
    fade_samples = int(fade_duration * sr)
    fade_samples = min(fade_samples, len(audio1), len(audio2))
    
    if fade_samples == 0:
        return np.concatenate([audio1, audio2])
    
    # Create fade curves
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)
    
    # Apply crossfade
    tail1 = audio1[-fade_samples:]
    head2 = audio2[:fade_samples]
    
    crossfaded = tail1 * fade_out + head2 * fade_in
    
    # Combine
    result = np.concatenate([
        audio1[:-fade_samples],
        crossfaded,
        audio2[fade_samples:]
    ])
    
    return result

