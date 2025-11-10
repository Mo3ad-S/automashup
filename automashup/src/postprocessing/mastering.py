"""
Mastering module for final processing and loudness normalization.
"""
import numpy as np
import pyloudnorm as pyln
from scipy import signal
from typing import Dict, Any, Optional


class Mastering:
    """Mastering module for final processing."""
    
    def __init__(self, sr: int = 44100, target_lufs: float = -14.0):
        self.sr = sr
        self.target_lufs = target_lufs
    
    def master(self, audio: np.ndarray, 
               apply_limiting: bool = True,
               apply_eq: bool = True,
               apply_stereo_enhancement: bool = False,
               normalize_loudness: bool = True) -> np.ndarray:
        """
        Master audio with final processing.
        
        Args:
            audio: Input audio
            apply_limiting: Whether to apply limiting
            apply_eq: Whether to apply final EQ
            apply_stereo_enhancement: Whether to enhance stereo
            normalize_loudness: Whether to normalize loudness
        
        Returns:
            Mastered audio
        """
        mastered = audio.copy()
        
        # Limiting
        if apply_limiting:
            mastered = self.apply_limiting(mastered)
        
        # Final EQ
        if apply_eq:
            mastered = self.apply_final_eq(mastered)
        
        # Stereo enhancement
        if apply_stereo_enhancement and len(mastered.shape) > 1:
            mastered = self.enhance_stereo(mastered)
        
        # Loudness normalization
        if normalize_loudness:
            mastered = self.normalize_loudness(mastered)
        
        return mastered
    
    def apply_limiting(self, audio: np.ndarray, 
                      threshold: float = 0.95,
                      release: float = 0.01) -> np.ndarray:
        """
        Apply limiting to prevent clipping.
        
        Args:
            audio: Input audio
            threshold: Limiting threshold
            release: Release time in seconds
        
        Returns:
            Limited audio
        """
        if len(audio) == 0:
            return audio
        
        # Soft limiting using tanh
        limited = np.tanh(audio * (1.0 / threshold)) * threshold
        
        # Hard limiting as backup
        limited = np.clip(limited, -0.99, 0.99)
        
        return limited
    
    def apply_final_eq(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply final EQ pass.
        
        Args:
            audio: Input audio
        
        Returns:
            EQ'd audio
        """
        if len(audio) == 0:
            return audio
        
        # Gentle high-pass filter to remove sub-bass
        nyquist = self.sr / 2
        b, a = signal.butter(2, 30 / nyquist, btype='high')
        eqd = signal.filtfilt(b, a, audio)
        
        # Gentle high-frequency boost
        b, a = signal.iirpeak(10000 / nyquist, 2.0)
        gain = 1.1
        b = b * gain
        eqd = signal.filtfilt(b, a, eqd)
        
        return eqd
    
    def enhance_stereo(self, audio: np.ndarray) -> np.ndarray:
        """
        Enhance stereo width.
        
        Args:
            audio: Input stereo audio
        
        Returns:
            Enhanced stereo audio
        """
        if len(audio.shape) < 2 or audio.shape[0] < 2:
            return audio
        
        # Extract channels
        left = audio[0, :]
        right = audio[1, :]
        
        # Calculate mid and side
        mid = (left + right) / 2
        side = (left - right) / 2
        
        # Enhance side slightly
        side_enhanced = side * 1.1
        
        # Reconstruct
        left_enhanced = mid + side_enhanced
        right_enhanced = mid - side_enhanced
        
        return np.stack([left_enhanced, right_enhanced])
    
    def normalize_loudness(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize loudness to target LUFS.
        
        Args:
            audio: Input audio
        
        Returns:
            Loudness-normalized audio
        """
        if len(audio) == 0:
            return audio
        
        try:
            # Create loudness meter
            meter = pyln.Meter(self.sr)
            
            # Measure loudness
            loudness = meter.integrated_loudness(audio)
            
            # Normalize to target
            normalized = pyln.normalize.loudness(audio, loudness, self.target_lufs)
            
            return normalized
        except Exception as e:
            print(f"Loudness normalization failed: {e}, using peak normalization")
            # Fallback to peak normalization
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                normalized = audio * (0.95 / max_val)
            else:
                normalized = audio
            return normalized
    
    def prevent_clipping(self, audio: np.ndarray) -> np.ndarray:
        """
        Prevent clipping.
        
        Args:
            audio: Input audio
        
        Returns:
            Audio with clipping prevented
        """
        if len(audio) == 0:
            return audio
        
        # Check for clipping
        max_val = np.max(np.abs(audio))
        
        if max_val > 0.99:
            # Soft limiting
            audio = np.tanh(audio * (0.99 / max_val)) * 0.99
        
        return audio

