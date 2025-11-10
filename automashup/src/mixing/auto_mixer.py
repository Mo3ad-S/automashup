"""
Automatic mixing engine with EQ, compression, and spatial processing.
"""
import numpy as np
import librosa
from scipy import signal
from typing import Dict, Any, List, Optional


class AutoMixer:
    """Automatic mixing engine."""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
    
    def mix(self, tracks: List[np.ndarray], track_types: List[str],
            apply_eq: bool = True, apply_compression: bool = True,
            apply_spatial: bool = True) -> np.ndarray:
        """
        Automatically mix multiple tracks.
        
        Args:
            tracks: List of audio tracks
            track_types: List of track types ('vocals', 'instrumental', etc.)
            apply_eq: Whether to apply automatic EQ
            apply_compression: Whether to apply compression
            apply_spatial: Whether to apply spatial processing
        
        Returns:
            Mixed audio
        """
        if not tracks:
            return np.array([])
        
        # Process each track
        processed_tracks = []
        for i, (track, track_type) in enumerate(zip(tracks, track_types)):
            processed = track.copy()
            
            # Automatic EQ
            if apply_eq:
                processed = self.apply_auto_eq(processed, track_type, self.sr)
            
            # Compression
            if apply_compression:
                processed = self.apply_compression(processed, track_type, self.sr)
            
            # Spatial processing
            if apply_spatial:
                processed = self.apply_spatial_processing(processed, track_type, self.sr)
            
            processed_tracks.append(processed)
        
        # Mix tracks
        mixed = self._mix_tracks(processed_tracks)
        
        # Gain staging
        mixed = self.apply_gain_staging(mixed)
        
        return mixed
    
    def apply_auto_eq(self, audio: np.ndarray, track_type: str, sr: int = 44100) -> np.ndarray:
        """
        Apply automatic EQ based on track type.
        
        Args:
            audio: Input audio
            track_type: Type of track
            sr: Sample rate
        
        Returns:
            EQ'd audio
        """
        if len(audio) == 0:
            return audio
        
        # Define EQ curves for different track types
        if track_type == 'vocals':
            # Boost mid frequencies for vocals
            audio = self._apply_parametric_eq(audio, sr, 
                                             frequency=2000, gain=2.0, q=2.0)
            audio = self._apply_parametric_eq(audio, sr,
                                             frequency=500, gain=1.5, q=1.5)
        elif track_type == 'bass':
            # Boost low frequencies for bass
            audio = self._apply_parametric_eq(audio, sr,
                                             frequency=100, gain=3.0, q=1.0)
        elif track_type == 'drums':
            # Boost low and high frequencies for drums
            audio = self._apply_parametric_eq(audio, sr,
                                             frequency=80, gain=2.0, q=1.0)
            audio = self._apply_parametric_eq(audio, sr,
                                             frequency=5000, gain=1.5, q=2.0)
        elif track_type == 'instrumental':
            # Gentle EQ for instrumental
            audio = self._apply_parametric_eq(audio, sr,
                                             frequency=1000, gain=1.2, q=1.5)
        
        return audio
    
    def _apply_parametric_eq(self, audio: np.ndarray, sr: int,
                            frequency: float, gain: float, q: float) -> np.ndarray:
        """Apply parametric EQ."""
        # Design parametric EQ filter
        nyquist = sr / 2
        normalized_freq = frequency / nyquist
        
        # Design biquad filter
        b, a = signal.iirpeak(normalized_freq, q)
        
        # Apply gain
        gain_linear = 10 ** (gain / 20.0)
        b = b * gain_linear
        
        # Apply filter
        filtered = signal.filtfilt(b, a, audio)
        
        return filtered
    
    def apply_compression(self, audio: np.ndarray, track_type: str, sr: int = 44100) -> np.ndarray:
        """
        Apply compression based on track type.
        
        Args:
            audio: Input audio
            track_type: Type of track
            sr: Sample rate
        
        Returns:
            Compressed audio
        """
        if len(audio) == 0:
            return audio
        
        # Compression parameters based on track type
        if track_type == 'vocals':
            threshold = 0.3
            ratio = 3.0
            attack = 0.003  # 3ms
            release = 0.1   # 100ms
        elif track_type == 'bass':
            threshold = 0.4
            ratio = 4.0
            attack = 0.005
            release = 0.15
        elif track_type == 'drums':
            threshold = 0.5
            ratio = 2.0
            attack = 0.001
            release = 0.05
        else:
            threshold = 0.4
            ratio = 2.5
            attack = 0.003
            release = 0.1
        
        # Apply compression
        compressed = self._apply_compressor(audio, threshold, ratio, attack, release, sr)
        
        return compressed
    
    def _apply_compressor(self, audio: np.ndarray, threshold: float,
                         ratio: float, attack: float, release: float, sr: int) -> np.ndarray:
        """Apply compressor."""
        # Simple compressor implementation
        compressed = audio.copy()
        
        # Calculate envelope
        envelope = np.abs(audio)
        
        # Smooth envelope (attack/release)
        alpha_attack = np.exp(-1.0 / (attack * sr))
        alpha_release = np.exp(-1.0 / (release * sr))
        
        smoothed_envelope = np.zeros_like(envelope)
        for i in range(1, len(envelope)):
            if envelope[i] > smoothed_envelope[i-1]:
                # Attack
                smoothed_envelope[i] = envelope[i] + alpha_attack * (smoothed_envelope[i-1] - envelope[i])
            else:
                # Release
                smoothed_envelope[i] = envelope[i] + alpha_release * (smoothed_envelope[i-1] - envelope[i])
        
        # Apply compression
        gain_reduction = np.ones_like(audio)
        above_threshold = smoothed_envelope > threshold
        
        if np.any(above_threshold):
            # Calculate gain reduction
            excess = smoothed_envelope[above_threshold] - threshold
            reduction = excess / ratio
            gain_reduction[above_threshold] = 1.0 - (reduction / smoothed_envelope[above_threshold])
        
        # Apply gain reduction
        compressed = compressed * gain_reduction
        
        return compressed
    
    def apply_spatial_processing(self, audio: np.ndarray, track_type: str, sr: int = 44100) -> np.ndarray:
        """
        Apply spatial processing (panning, reverb, delay).
        
        Args:
            audio: Input audio
            track_type: Type of track
            sr: Sample rate
        
        Returns:
            Spatially processed audio
        """
        if len(audio) == 0:
            return audio
        
        # Panning based on track type
        if track_type == 'vocals':
            # Center vocals
            panned = audio
        elif track_type == 'bass':
            # Center bass
            panned = audio
        elif track_type == 'drums':
            # Slight stereo spread for drums
            panned = self._apply_stereo_spread(audio, 0.1)
        else:
            # Slight stereo spread for other instruments
            panned = self._apply_stereo_spread(audio, 0.05)
        
        # Add subtle reverb (simplified)
        panned = self._apply_reverb(panned, sr, room_size=0.1, wet=0.1)
        
        return panned
    
    def _apply_stereo_spread(self, audio: np.ndarray, spread: float) -> np.ndarray:
        """Apply stereo spread."""
        # Convert to stereo if mono
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio])
        
        # Apply stereo spread
        mid = (audio[0] + audio[1]) / 2
        side = (audio[0] - audio[1]) / 2
        
        side_spread = side * (1 + spread)
        
        left = mid + side_spread
        right = mid - side_spread
        
        return np.stack([left, right])
    
    def _apply_reverb(self, audio: np.ndarray, sr: int, room_size: float, wet: float) -> np.ndarray:
        """Apply simple reverb."""
        # Simple reverb using delay and feedback
        delay_samples = int(room_size * sr)
        
        if delay_samples > 0 and delay_samples < len(audio):
            delayed = np.pad(audio, (delay_samples, 0), mode='constant')[:len(audio)]
            reverb = audio + wet * delayed * 0.5
            return reverb
        
        return audio
    
    def apply_gain_staging(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply intelligent gain staging.
        
        Args:
            audio: Input audio
        
        Returns:
            Gain-staged audio
        """
        if len(audio) == 0:
            return audio
        
        # Calculate RMS
        rms = np.sqrt(np.mean(audio**2))
        
        # Target RMS
        target_rms = 0.3
        
        # Calculate gain
        if rms > 0:
            gain = target_rms / rms
            # Limit gain
            gain = min(gain, 2.0)
        else:
            gain = 1.0
        
        # Apply gain
        staged = audio * gain
        
        # Prevent clipping
        max_val = np.max(np.abs(staged))
        if max_val > 0.95:
            staged = staged * 0.95 / max_val
        
        return staged
    
    def _mix_tracks(self, tracks: List[np.ndarray]) -> np.ndarray:
        """Mix multiple tracks together."""
        if not tracks:
            return np.array([])
        
        # Find maximum length
        max_length = max(len(track) for track in tracks)
        
        # Pad all tracks to same length
        padded_tracks = []
        for track in tracks:
            if len(track) < max_length:
                padded = np.pad(track, (0, max_length - len(track)), mode='constant')
            else:
                padded = track[:max_length]
            padded_tracks.append(padded)
        
        # Sum tracks
        mixed = np.sum(padded_tracks, axis=0)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 0.95:
            mixed = mixed * 0.95 / max_val
        
        return mixed

