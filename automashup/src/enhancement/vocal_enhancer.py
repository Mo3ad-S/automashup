"""
Vocal-specific enhancement module.
"""
import numpy as np
import librosa
from typing import Dict, Any


class VocalEnhancer:
    """Vocal-specific enhancement."""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
    
    def enhance_vocals(self, audio: np.ndarray, sr: int = 44100,
                      preserve_formants: bool = True,
                      deess: bool = True,
                      reduce_breath: bool = True,
                      enhance_intelligibility: bool = True) -> np.ndarray:
        """
        Enhance vocal audio.
        
        Args:
            audio: Input vocal audio
            sr: Sample rate
            preserve_formants: Whether to preserve formants
            deess: Whether to reduce sibilance
            reduce_breath: Whether to reduce breath noise
            enhance_intelligibility: Whether to enhance intelligibility
        
        Returns:
            Enhanced vocal audio
        """
        enhanced = audio.copy()
        
        # Formant preservation (already done in pitch correction)
        if preserve_formants:
            enhanced = self._preserve_formants(enhanced, sr)
        
        # De-essing
        if deess:
            enhanced = self.deess(enhanced, sr)
        
        # Breath noise reduction
        if reduce_breath:
            enhanced = self.reduce_breath_noise(enhanced, sr)
        
        # Intelligibility enhancement
        if enhance_intelligibility:
            enhanced = self.enhance_intelligibility(enhanced, sr)
        
        return enhanced
    
    def _preserve_formants(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Preserve formants (placeholder - done in pitch correction)."""
        # Formant preservation is handled in pitch_corrector
        return audio
    
    def deess(self, audio: np.ndarray, sr: int = 44100) -> np.ndarray:
        """
        Reduce sibilance (de-essing).
        
        Args:
            audio: Input audio
            sr: Sample rate
        
        Returns:
            De-essed audio
        """
        if len(audio) == 0:
            return audio
        
        # Detect sibilant frequencies (4-8 kHz)
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        frequencies = librosa.fft_frequencies(sr=sr)
        
        # Find sibilant frequency range
        sibilant_mask = (frequencies >= 4000) & (frequencies <= 8000)
        
        # Reduce sibilant frequencies
        magnitude_sibilant = magnitude.copy()
        magnitude_sibilant[sibilant_mask, :] *= 0.7  # Reduce by 30%
        
        # Reconstruct
        enhanced_stft = magnitude_sibilant * np.exp(1j * phase)
        enhanced = librosa.istft(enhanced_stft)
        
        return enhanced
    
    def reduce_breath_noise(self, audio: np.ndarray, sr: int = 44100) -> np.ndarray:
        """
        Reduce breath noise.
        
        Args:
            audio: Input audio
            sr: Sample rate
        
        Returns:
            Audio with reduced breath noise
        """
        if len(audio) == 0:
            return audio
        
        # Detect breath noise (low frequency, high energy)
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        frequencies = librosa.fft_frequencies(sr=sr)
        
        # Find breath noise frequency range (100-500 Hz)
        breath_mask = (frequencies >= 100) & (frequencies <= 500)
        
        # Reduce breath noise
        magnitude_breath = magnitude.copy()
        # Only reduce if energy is high (likely breath)
        breath_energy = np.mean(magnitude[breath_mask, :], axis=0)
        breath_threshold = np.percentile(breath_energy, 75)
        
        for t in range(magnitude.shape[1]):
            if breath_energy[t] > breath_threshold:
                magnitude_breath[breath_mask, t] *= 0.6  # Reduce by 40%
        
        # Reconstruct
        enhanced_stft = magnitude_breath * np.exp(1j * phase)
        enhanced = librosa.istft(enhanced_stft)
        
        return enhanced
    
    def enhance_intelligibility(self, audio: np.ndarray, sr: int = 44100) -> np.ndarray:
        """
        Enhance vocal intelligibility.
        
        Args:
            audio: Input audio
            sr: Sample rate
        
        Returns:
            Audio with enhanced intelligibility
        """
        if len(audio) == 0:
            return audio
        
        # Boost speech formant frequencies (500-3000 Hz)
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        frequencies = librosa.fft_frequencies(sr=sr)
        
        # Find speech formant range
        formant_mask = (frequencies >= 500) & (frequencies <= 3000)
        
        # Boost formants
        magnitude_formant = magnitude.copy()
        magnitude_formant[formant_mask, :] *= 1.15  # Boost by 15%
        
        # Reconstruct
        enhanced_stft = magnitude_formant * np.exp(1j * phase)
        enhanced = librosa.istft(enhanced_stft)
        
        return enhanced

