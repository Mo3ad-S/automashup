"""
Silence detection and analysis module.
"""
import numpy as np
import librosa
from typing import List, Tuple, Dict, Any


class SilenceDetector:
    """Detect and classify silence regions in audio."""
    
    def __init__(self, sr: int = 44100, frame_length: int = 2048, hop_length: int = 512):
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
    
    def detect_silences(self, audio: np.ndarray, 
                        threshold_db: float = -40.0,
                        min_silence_duration: float = 0.1) -> List[Tuple[float, float]]:
        """
        Detect silence regions in audio.
        
        Args:
            audio: Audio signal
            threshold_db: Threshold in dB below which audio is considered silence
            min_silence_duration: Minimum duration in seconds for a silence region
        
        Returns:
            List of (start, end) tuples in seconds
        """
        if len(audio) == 0:
            return []
        
        # Calculate RMS energy per frame
        rms = librosa.feature.rms(y=audio, frame_length=self.frame_length, 
                                   hop_length=self.hop_length)[0]
        
        # Convert to dB
        rms_db = librosa.power_to_db(rms**2, ref=np.max(rms**2))
        
        # Find frames below threshold
        silence_frames = rms_db < threshold_db
        
        # Convert frame indices to time
        times = librosa.frames_to_time(np.arange(len(silence_frames)), 
                                      sr=self.sr, hop_length=self.hop_length)
        
        # Find continuous silence regions
        silence_regions = []
        in_silence = False
        silence_start = 0.0
        
        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_silence:
                # Start of silence
                in_silence = True
                silence_start = times[i]
            elif not is_silent and in_silence:
                # End of silence
                in_silence = False
                silence_end = times[i]
                duration = silence_end - silence_start
                if duration >= min_silence_duration:
                    silence_regions.append((silence_start, silence_end))
        
        # Handle silence that extends to end of audio
        if in_silence:
            silence_end = times[-1]
            duration = silence_end - silence_start
            if duration >= min_silence_duration:
                silence_regions.append((silence_start, silence_end))
        
        return silence_regions
    
    def classify_silence(self, silence_regions: List[Tuple[float, float]], 
                         audio_duration: float) -> List[Dict[str, Any]]:
        """
        Classify silence regions by type.
        
        Args:
            silence_regions: List of (start, end) tuples
            audio_duration: Total duration of audio in seconds
        
        Returns:
            List of dictionaries with silence information
        """
        classified = []
        
        for start, end in silence_regions:
            duration = end - start
            silence_type = self._classify_silence_type(start, end, duration, audio_duration)
            
            classified.append({
                'start': start,
                'end': end,
                'duration': duration,
                'type': silence_type,
                'relative_position': start / audio_duration if audio_duration > 0 else 0.0
            })
        
        return classified
    
    def _classify_silence_type(self, start: float, end: float, 
                               duration: float, total_duration: float) -> str:
        """Classify the type of silence."""
        # Short gaps (< 0.5s) - likely transitions or brief pauses
        if duration < 0.5:
            return 'short_gap'
        
        # Medium gaps (0.5-2s) - likely missing segments or transitions
        elif duration < 2.0:
            return 'medium_gap'
        
        # Long gaps (> 2s) - likely missing segments
        else:
            return 'long_gap'
    
    def detect_transitions(self, audio: np.ndarray, 
                          silence_regions: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """
        Detect transitions between segments (potential silence regions).
        
        Args:
            audio: Audio signal
            silence_regions: Previously detected silence regions
        
        Returns:
            List of transition information
        """
        transitions = []
        
        # Analyze regions around detected silences for transition characteristics
        for start, end in silence_regions:
            # Get context before and after silence
            context_before = max(0, start - 0.5)
            context_after = min(len(audio) / self.sr, end + 0.5)
            
            # Extract audio around transition
            before_start = int(context_before * self.sr)
            before_end = int(start * self.sr)
            after_start = int(end * self.sr)
            after_end = int(context_after * self.sr)
            
            if before_end > before_start and after_end > after_start:
                before_audio = audio[before_start:before_end]
                after_audio = audio[after_start:after_end]
                
                # Calculate energy difference
                energy_before = np.mean(before_audio**2)
                energy_after = np.mean(after_audio**2)
                energy_diff = abs(energy_before - energy_after) / max(energy_before, energy_after, 1e-10)
                
                transitions.append({
                    'start': start,
                    'end': end,
                    'energy_diff': energy_diff,
                    'is_transition': energy_diff > 0.3  # Significant energy change
                })
        
        return transitions

