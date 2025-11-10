"""
Audio enhancement pipeline for denoising, de-artifacting, and quality restoration.
"""
import numpy as np
import librosa
from typing import Dict, Any, Optional
try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False


class AudioEnhancer:
    """Audio enhancement pipeline."""
    
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
    
    def enhance(self, audio: np.ndarray, sr: int = 44100, 
                denoise: bool = True, deartifact: bool = True,
                restore_harmonics: bool = True) -> np.ndarray:
        """
        Enhance audio quality.
        
        Args:
            audio: Input audio
            sr: Sample rate
            denoise: Whether to denoise
            deartifact: Whether to remove artifacts
            restore_harmonics: Whether to restore harmonics
        
        Returns:
            Enhanced audio
        """
        enhanced = audio.copy()
        
        # Denoising
        if denoise:
            enhanced = self.denoise(enhanced, sr)
        
        # De-artifacting
        if deartifact:
            enhanced = self.deartifact(enhanced, sr)
        
        # Harmonic restoration
        if restore_harmonics:
            enhanced = self.restore_harmonics(enhanced, sr)
        
        # Dynamic range restoration
        enhanced = self.restore_dynamic_range(enhanced)
        
        return enhanced
    
    def denoise(self, audio: np.ndarray, sr: int = 44100) -> np.ndarray:
        """
        Denoise audio.
        
        Args:
            audio: Input audio
            sr: Sample rate
        
        Returns:
            Denoised audio
        """
        if len(audio) == 0:
            return audio
        
        if HAS_NOISEREDUCE:
            # Use noisereduce library
            try:
                denoised = nr.reduce_noise(y=audio, sr=sr)
                return denoised
            except Exception as e:
                print(f"Noisereduce failed: {e}, using spectral subtraction")
                return self._spectral_subtraction(audio, sr)
        else:
            # Fallback to spectral subtraction
            return self._spectral_subtraction(audio, sr)
    
    def _spectral_subtraction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Spectral subtraction denoising."""
        # Simple spectral subtraction
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise floor (from first few frames)
        noise_floor = np.mean(magnitude[:, :10], axis=1, keepdims=True)
        
        # Subtract noise
        enhanced_magnitude = magnitude - 0.5 * noise_floor
        enhanced_magnitude = np.maximum(enhanced_magnitude, 0.1 * magnitude)
        
        # Reconstruct
        enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
        enhanced = librosa.istft(enhanced_stft)
        
        return enhanced
    
    def deartifact(self, audio: np.ndarray, sr: int = 44100) -> np.ndarray:
        """
        Remove artifacts from time-stretch/pitch-shift operations.
        
        Args:
            audio: Input audio
            sr: Sample rate
        
        Returns:
            De-artifacted audio
        """
        if len(audio) == 0:
            return audio
        
        # Detect and remove artifacts
        # 1. Remove sudden discontinuities
        enhanced = self._remove_discontinuities(audio)
        
        # 2. Smooth phase vocoder artifacts
        enhanced = self._smooth_phase_artifacts(enhanced, sr)
        
        # 3. Remove spectral artifacts
        enhanced = self._remove_spectral_artifacts(enhanced, sr)
        
        return enhanced
    
    def _remove_discontinuities(self, audio: np.ndarray) -> np.ndarray:
        """Remove sudden discontinuities."""
        # Detect large jumps
        diff = np.diff(audio)
        threshold = np.std(diff) * 3
        large_jumps = np.abs(diff) > threshold
        
        # Smooth large jumps
        if np.any(large_jumps):
            smoothed = audio.copy()
            for i in np.where(large_jumps)[0]:
                # Smooth transition
                window_size = min(10, len(audio) - i - 1)
                if window_size > 0:
                    smoothed[i:i+window_size] = np.linspace(
                        audio[i],
                        audio[i+window_size],
                        window_size
                    )
            return smoothed
        
        return audio
    
    def _smooth_phase_artifacts(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Smooth phase vocoder artifacts."""
        # Apply gentle low-pass filter to reduce phase artifacts
        from scipy import signal
        
        # Design low-pass filter
        nyquist = sr / 2
        cutoff = nyquist * 0.95  # Keep most frequencies
        b, a = signal.butter(4, cutoff / nyquist, btype='low')
        
        # Apply filter
        filtered = signal.filtfilt(b, a, audio)
        
        return filtered
    
    def _remove_spectral_artifacts(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Remove spectral artifacts."""
        # Use spectral gating to remove artifacts
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Detect artifacts (unusual spectral patterns)
        magnitude_mean = np.mean(magnitude, axis=1, keepdims=True)
        magnitude_std = np.std(magnitude, axis=1, keepdims=True)
        
        # Gate out outliers
        threshold = magnitude_mean + 2 * magnitude_std
        magnitude_gated = np.minimum(magnitude, threshold)
        
        # Reconstruct
        enhanced_stft = magnitude_gated * np.exp(1j * phase)
        enhanced = librosa.istft(enhanced_stft)
        
        return enhanced
    
    def restore_harmonics(self, audio: np.ndarray, sr: int = 44100) -> np.ndarray:
        """
        Restore lost harmonics.
        
        Args:
            audio: Input audio
            sr: Sample rate
        
        Returns:
            Audio with restored harmonics
        """
        if len(audio) == 0:
            return audio
        
        # Use harmonic-percussive separation to enhance harmonics
        harmonic, percussive = librosa.effects.hpss(audio)
        
        # Enhance harmonics
        stft_harmonic = librosa.stft(harmonic)
        magnitude = np.abs(stft_harmonic)
        phase = np.angle(stft_harmonic)
        
        # Boost harmonics (simplified)
        magnitude_enhanced = magnitude * 1.1  # Slight boost
        
        # Reconstruct
        enhanced_stft = magnitude_enhanced * np.exp(1j * phase)
        enhanced_harmonic = librosa.istft(enhanced_stft)
        
        # Combine with percussive
        enhanced = enhanced_harmonic + percussive * 0.9
        
        return enhanced
    
    def restore_dynamic_range(self, audio: np.ndarray) -> np.ndarray:
        """
        Restore dynamic range.
        
        Args:
            audio: Input audio
        
        Returns:
            Audio with restored dynamic range
        """
        if len(audio) == 0:
            return audio
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:
            # Soft limiting
            audio = np.tanh(audio * 0.95 / max_val)
        
        return audio

