"""
Silence handling and analysis for audio inpainting.
"""
import numpy as np
from typing import List, Dict, Any, Tuple
from automashup.src.core.track import Track
from automashup.src.core.segment import Segment
from automashup.src.preprocessing.silence_detector import SilenceDetector


class SilenceHandler:
    """Handle silence regions and prepare for inpainting."""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.silence_detector = SilenceDetector(sr=sr)
    
    def detect_all_silences(self, track: Track) -> List[Dict[str, Any]]:
        """
        Detect all silence regions in track.
        
        Args:
            track: Track to analyze
        
        Returns:
            List of silence information dictionaries
        """
        # Detect silences in full track
        silence_regions = self.silence_detector.detect_silences(track.audio)
        
        # Classify silences
        audio_duration = len(track.audio) / track.sr
        classified_silences = self.silence_detector.classify_silence(
            silence_regions,
            audio_duration
        )
        
        # Add to track
        for silence in classified_silences:
            track.add_silence_region(
                silence['start'],
                silence['end'],
                silence['type']
            )
        
        return classified_silences
    
    def analyze_silence_context(self, audio: np.ndarray, 
                                silence_start: float, silence_end: float,
                                context_duration: float = 1.0) -> Dict[str, Any]:
        """
        Analyze context around silence for inpainting.
        
        Args:
            audio: Full audio signal
            silence_start: Start time of silence
            silence_end: End time of silence
            context_duration: Duration of context to analyze
        
        Returns:
            Dictionary with context features
        """
        sr = self.sr
        silence_duration = silence_end - silence_start
        
        # Get context before and after
        context_before_start = max(0, silence_start - context_duration)
        context_before_end = silence_start
        context_after_start = silence_end
        context_after_end = min(len(audio) / sr, silence_end + context_duration)
        
        # Extract context audio
        before_start_idx = int(context_before_start * sr)
        before_end_idx = int(context_before_end * sr)
        after_start_idx = int(context_after_start * sr)
        after_end_idx = int(context_after_end * sr)
        
        context_before = audio[before_start_idx:before_end_idx] if before_end_idx > before_start_idx else np.array([])
        context_after = audio[after_start_idx:after_end_idx] if after_end_idx > after_start_idx else np.array([])
        
        # Extract features
        features = {
            'silence_duration': silence_duration,
            'silence_type': self._classify_silence_type(silence_duration),
            'context_before': context_before,
            'context_after': context_after,
            'has_context_before': len(context_before) > 0,
            'has_context_after': len(context_after) > 0
        }
        
        # Analyze spectral features if context available
        if len(context_before) > 0:
            features['spectral_features_before'] = self._extract_spectral_features(context_before, sr)
        if len(context_after) > 0:
            features['spectral_features_after'] = self._extract_spectral_features(context_after, sr)
        
        return features
    
    def _classify_silence_type(self, duration: float) -> str:
        """Classify silence type by duration."""
        if duration < 0.5:
            return 'short_gap'
        elif duration < 2.0:
            return 'medium_gap'
        else:
            return 'long_gap'
    
    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract spectral features from audio."""
        import librosa
        
        if len(audio) == 0:
            return {}
        
        # Spectral centroid
        stft = librosa.stft(audio)
        spectral_centroid = librosa.feature.spectral_centroid(S=np.abs(stft))[0]
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        
        return {
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'chroma_mean': np.mean(chroma, axis=1),
            'mfcc_mean': np.mean(mfcc, axis=1)
        }
    
    def prepare_inpainting_regions(self, track: Track) -> List[Dict[str, Any]]:
        """
        Prepare all silence regions for inpainting.
        
        Args:
            track: Track to process
        
        Returns:
            List of inpainting region dictionaries
        """
        # Detect all silences
        silences = self.detect_all_silences(track)
        
        # Prepare each silence for inpainting
        inpainting_regions = []
        
        for silence in silences:
            context = self.analyze_silence_context(
                track.audio,
                silence['start'],
                silence['end']
            )
            
            inpainting_regions.append({
                'start': silence['start'],
                'end': silence['end'],
                'duration': silence['duration'],
                'type': silence['type'],
                'context': context,
                'start_samples': int(silence['start'] * track.sr),
                'end_samples': int(silence['end'] * track.sr)
            })
        
        return inpainting_regions

