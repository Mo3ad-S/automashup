"""
Stem mixing module for separate vocal and instrumental processing.
"""
import numpy as np
from typing import List, Dict, Any
from automashup.src.core.track import Track
from automashup.src.mixing.auto_mixer import AutoMixer


class StemMixer:
    """Stem mixer for separate vocal and instrumental processing."""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.auto_mixer = AutoMixer(sr=sr)
    
    def mix_stems(self, tracks: List[Track], 
                  apply_ducking: bool = True,
                  apply_separate_processing: bool = True) -> np.ndarray:
        """
        Mix stems with separate processing for vocals and instrumental.
        
        Args:
            tracks: List of Track objects
            apply_ducking: Whether to apply sidechain ducking
            apply_separate_processing: Whether to apply separate processing chains
        
        Returns:
            Mixed audio
        """
        # Separate vocals and instrumental
        vocals = []
        instrumental = []
        track_types = []
        
        for track in tracks:
            if track.track_type == 'vocals':
                vocals.append(track.audio)
                track_types.append('vocals')
            else:
                instrumental.append(track.audio)
                track_types.append(track.track_type)
        
        # Process vocals and instrumental separately
        if apply_separate_processing:
            if vocals:
                vocals_processed = self._process_vocals(vocals)
            else:
                vocals_processed = []
            
            if instrumental:
                # Get track types for instrumental tracks
                inst_track_types = [t for t in track_types if t != 'vocals']
                instrumental_processed = self._process_instrumental(instrumental, inst_track_types)
            else:
                instrumental_processed = []
        else:
            vocals_processed = vocals
            instrumental_processed = instrumental
        
        # Apply ducking (sidechain vocals to instrumental)
        if apply_ducking and vocals_processed and instrumental_processed:
            vocals_processed = self._apply_ducking(vocals_processed, instrumental_processed)
        
        # Mix all stems
        all_stems = vocals_processed + instrumental_processed
        mixed = self.auto_mixer._mix_tracks(all_stems)
        
        return mixed
    
    def _process_vocals(self, vocals: List[np.ndarray]) -> List[np.ndarray]:
        """Process vocal stems."""
        processed = []
        
        for vocal in vocals:
            # Apply vocal-specific processing
            processed_vocal = self.auto_mixer.apply_auto_eq(vocal, 'vocals', self.sr)
            processed_vocal = self.auto_mixer.apply_compression(processed_vocal, 'vocals', self.sr)
            processed_vocal = self.auto_mixer.apply_spatial_processing(processed_vocal, 'vocals', self.sr)
            processed.append(processed_vocal)
        
        return processed
    
    def _process_instrumental(self, instrumental: List[np.ndarray], 
                             track_types: List[str] = None) -> List[np.ndarray]:
        """Process instrumental stems."""
        processed = []
        
        if track_types is None:
            track_types = ['instrumental'] * len(instrumental)
        
        for inst, track_type in zip(instrumental, track_types):
            # Apply instrumental-specific processing
            processed_inst = self.auto_mixer.apply_auto_eq(inst, track_type, self.sr)
            processed_inst = self.auto_mixer.apply_compression(processed_inst, track_type, self.sr)
            processed_inst = self.auto_mixer.apply_spatial_processing(processed_inst, track_type, self.sr)
            processed.append(processed_inst)
        
        return processed
    
    def _apply_ducking(self, vocals: List[np.ndarray], 
                      instrumental: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply sidechain ducking (reduce instrumental when vocals are present).
        
        Args:
            vocals: List of vocal tracks
            instrumental: List of instrumental tracks
        
        Returns:
            List of ducked vocal tracks
        """
        # Sum vocals to get vocal envelope
        if not vocals:
            return vocals
        
        # Find max length
        max_length = max(len(v) for v in vocals)
        
        # Sum vocals
        vocal_sum = np.zeros(max_length)
        for vocal in vocals:
            padded = np.pad(vocal, (0, max_length - len(vocal)), mode='constant')
            vocal_sum += padded
        
        # Calculate vocal envelope
        vocal_envelope = np.abs(vocal_sum)
        
        # Smooth envelope
        from scipy import signal
        alpha = 0.1
        smoothed_envelope = signal.lfilter([alpha], [1, alpha-1], vocal_envelope)
        
        # Normalize envelope
        if np.max(smoothed_envelope) > 0:
            smoothed_envelope = smoothed_envelope / np.max(smoothed_envelope)
        
        # Apply ducking to vocals (boost when vocals are present)
        ducked_vocals = []
        for vocal in vocals:
            padded = np.pad(vocal, (0, max_length - len(vocal)), mode='constant')
            
            # Boost vocals when they're present
            vocal_boost = 1.0 + 0.2 * smoothed_envelope  # 20% boost
            ducked = padded * vocal_boost
            
            # Trim back to original length
            ducked = ducked[:len(vocal)]
            ducked_vocals.append(ducked)
        
        return ducked_vocals

