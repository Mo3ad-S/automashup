"""
Transition smoothing module for crossfades and volume automation.
"""
import numpy as np
import librosa
from typing import List, Tuple, Dict, Any


class TransitionSmoother:
    """Smooth transitions between segments."""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
    
    def smooth_transitions(self, audio: np.ndarray, 
                          transition_points: List[float],
                          crossfade_duration: float = 0.1) -> np.ndarray:
        """
        Smooth transitions at specified points.
        
        Args:
            audio: Input audio
            transition_points: List of transition times in seconds
            crossfade_duration: Duration of crossfade in seconds
        
        Returns:
            Audio with smoothed transitions
        """
        if not transition_points:
            return audio
        
        smoothed = audio.copy()
        crossfade_samples = int(crossfade_duration * self.sr)
        
        for transition_time in transition_points:
            transition_idx = int(transition_time * self.sr)
            
            if transition_idx > crossfade_samples and transition_idx < len(audio) - crossfade_samples:
                # Apply crossfade
                smoothed = self._apply_crossfade(
                    smoothed,
                    transition_idx,
                    crossfade_samples
                )
        
        return smoothed
    
    def _apply_crossfade(self, audio: np.ndarray, 
                        transition_idx: int, 
                        crossfade_samples: int) -> np.ndarray:
        """Apply crossfade at transition point."""
        # Create fade curves
        fade_out = np.linspace(1.0, 0.0, crossfade_samples)
        fade_in = np.linspace(0.0, 1.0, crossfade_samples)
        
        # Get segments
        before = audio[transition_idx - crossfade_samples:transition_idx]
        after = audio[transition_idx:transition_idx + crossfade_samples]
        
        # Apply fades
        before_faded = before * fade_out
        after_faded = after * fade_in
        
        # Crossfade
        crossfaded = before_faded + after_faded
        
        # Replace in audio
        smoothed = audio.copy()
        smoothed[transition_idx - crossfade_samples:transition_idx] = before_faded
        smoothed[transition_idx:transition_idx + crossfade_samples] = after_faded
        
        return smoothed
    
    def optimize_crossfade(self, audio1: np.ndarray, audio2: np.ndarray,
                           sr: int = 44100, 
                           crossfade_duration: float = 0.1) -> np.ndarray:
        """
        Optimize crossfade between two audio segments.
        
        Args:
            audio1: First audio segment
            audio2: Second audio segment
            sr: Sample rate
            crossfade_duration: Crossfade duration in seconds
        
        Returns:
            Crossfaded audio
        """
        crossfade_samples = int(crossfade_duration * sr)
        crossfade_samples = min(crossfade_samples, len(audio1), len(audio2))
        
        if crossfade_samples == 0:
            return np.concatenate([audio1, audio2])
        
        # Get segments for crossfade
        tail1 = audio1[-crossfade_samples:]
        head2 = audio2[:crossfade_samples]
        
        # Create optimized fade curves (smooth curves)
        fade_out = self._smooth_fade_curve(np.linspace(1.0, 0.0, crossfade_samples))
        fade_in = self._smooth_fade_curve(np.linspace(0.0, 1.0, crossfade_samples))
        
        # Apply crossfade
        crossfaded = tail1 * fade_out + head2 * fade_in
        
        # Combine
        result = np.concatenate([
            audio1[:-crossfade_samples],
            crossfaded,
            audio2[crossfade_samples:]
        ])
        
        return result
    
    def _smooth_fade_curve(self, linear_fade: np.ndarray) -> np.ndarray:
        """Create smooth fade curve (S-curve)."""
        # Apply S-curve for smoother fade
        smoothed = 3 * linear_fade**2 - 2 * linear_fade**3
        return smoothed
    
    def apply_volume_automation(self, audio: np.ndarray,
                               automation_points: List[Tuple[float, float]]) -> np.ndarray:
        """
        Apply volume automation.
        
        Args:
            audio: Input audio
            automation_points: List of (time, volume) tuples
        
        Returns:
            Audio with volume automation
        """
        if not automation_points:
            return audio
        
        # Create automation curve
        automation_curve = np.ones(len(audio))
        
        for i, (time, volume) in enumerate(automation_points):
            idx = int(time * self.sr)
            
            if i == 0:
                # First point: set volume from start
                automation_curve[:idx] = volume
            else:
                # Interpolate between points
                prev_time, prev_volume = automation_points[i-1]
                prev_idx = int(prev_time * self.sr)
                
                # Linear interpolation
                n_samples = idx - prev_idx
                if n_samples > 0:
                    automation_curve[prev_idx:idx] = np.linspace(
                        prev_volume,
                        volume,
                        n_samples
                    )
        
        # Apply automation
        automated = audio * automation_curve
        
        return automated
    
    def smooth_frequency_transitions(self, audio: np.ndarray,
                                    transition_points: List[float],
                                    sr: int = 44100) -> np.ndarray:
        """
        Smooth frequency transitions at segment boundaries.
        
        Args:
            audio: Input audio
            transition_points: List of transition times
            sr: Sample rate
        
        Returns:
            Audio with smoothed frequency transitions
        """
        if not transition_points:
            return audio
        
        smoothed = audio.copy()
        
        for transition_time in transition_points:
            transition_idx = int(transition_time * sr)
            
            # Analyze frequency content before and after
            window_size = int(0.1 * sr)  # 100ms window
            
            if transition_idx > window_size and transition_idx < len(audio) - window_size:
                before = audio[transition_idx - window_size:transition_idx]
                after = audio[transition_idx:transition_idx + window_size]
                
                # Calculate spectral centroids
                centroid_before = np.mean(librosa.feature.spectral_centroid(
                    y=before, sr=sr)[0])
                centroid_after = np.mean(librosa.feature.spectral_centroid(
                    y=after, sr=sr)[0])
                
                # Smooth transition if large difference
                if abs(centroid_before - centroid_after) > 500:
                    # Apply gentle low-pass filter at transition
                    from scipy import signal
                    nyquist = sr / 2
                    b, a = signal.butter(2, 0.8, btype='low')
                    
                    transition_region = audio[transition_idx - window_size:transition_idx + window_size]
                    filtered = signal.filtfilt(b, a, transition_region)
                    
                    smoothed[transition_idx - window_size:transition_idx + window_size] = filtered
        
        return smoothed

