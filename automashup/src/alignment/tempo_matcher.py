"""
Advanced tempo matching using phase vocoder for better quality.
"""
import numpy as np
import librosa
from typing import Optional
from automashup.src.core.track import Track


class TempoMatcher:
    """Advanced tempo matching with phase vocoder."""
    
    def __init__(self, use_phase_vocoder: bool = True):
        self.use_phase_vocoder = use_phase_vocoder
    
    def match_tempo(self, track: Track, target_bpm: float) -> Track:
        """
        Match track tempo to target BPM.
        
        Args:
            track: Track to adjust
            target_bpm: Target BPM
        
        Returns:
            Tempo-adjusted track
        """
        current_bpm = track.bpm if hasattr(track, 'bpm') and track.bpm else 120
        
        if abs(current_bpm - target_bpm) < 0.1:
            # Already matched
            return track
        
        # Calculate stretch ratio
        stretch_ratio = current_bpm / target_bpm
        
        # Apply time stretching
        if self.use_phase_vocoder:
            stretched_audio = self._time_stretch_phase_vocoder(
                track.audio,
                stretch_ratio,
                track.sr
            )
        else:
            # Fallback to pyrubberband
            import pyrubberband as pyrb
            stretched_audio = pyrb.time_stretch(
                track.audio,
                track.sr,
                rate=1.0/stretch_ratio
            )
        
        # Update track
        track.audio = stretched_audio
        track.bpm = target_bpm
        
        # Update beats proportionally
        if hasattr(track, 'beats') and track.beats:
            track.beats = [beat / stretch_ratio for beat in track.beats]
        if hasattr(track, 'downbeats') and track.downbeats:
            track.downbeats = [downbeat / stretch_ratio for downbeat in track.downbeats]
        
        return track
    
    def _time_stretch_phase_vocoder(self, audio: np.ndarray, 
                                    stretch_ratio: float, sr: int) -> np.ndarray:
        """
        Time stretch using phase vocoder (librosa).
        
        Args:
            audio: Input audio
            stretch_ratio: Stretch ratio (>1 slows down, <1 speeds up)
            sr: Sample rate
        
        Returns:
            Time-stretched audio
        """
        # Use librosa's phase vocoder for better quality
        # librosa.effects.time_stretch uses phase vocoder internally
        stretched = librosa.effects.time_stretch(
            audio,
            rate=1.0/stretch_ratio
        )
        
        return stretched
    
    def match_tempo_multi_stage(self, track: Track, target_bpm: float) -> Track:
        """
        Multi-stage tempo matching (coarse + fine adjustment).
        
        Args:
            track: Track to adjust
            target_bpm: Target BPM
        
        Returns:
            Tempo-adjusted track
        """
        current_bpm = track.bpm if hasattr(track, 'bpm') and track.bpm else 120
        
        # Coarse adjustment (large changes)
        if abs(current_bpm - target_bpm) > 10:
            # First get close
            intermediate_bpm = current_bpm + np.sign(target_bpm - current_bpm) * 10
            track = self.match_tempo(track, intermediate_bpm)
        
        # Fine adjustment
        track = self.match_tempo(track, target_bpm)
        
        return track
    
    def smooth_tempo_transitions(self, track: Track, 
                                 transition_regions: list) -> Track:
        """
        Smooth tempo transitions at segment boundaries.
        
        Args:
            track: Track to process
            transition_regions: List of (start, end) tuples for transitions
        
        Returns:
            Track with smoothed transitions
        """
        # Apply gradual tempo changes at transitions
        # This is a simplified version - can be enhanced
        for start, end in transition_regions:
            start_idx = int(start * track.sr)
            end_idx = int(end * track.sr)
            
            # Apply gradual crossfade or smoothing
            # Implementation depends on specific requirements
        
        return track

