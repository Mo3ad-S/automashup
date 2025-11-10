"""
Quality assessment module for source separation and audio quality evaluation.
"""
import numpy as np
import librosa
from typing import Dict, Any, Optional


class QualityAssessment:
    """Assess quality of separated audio sources."""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
    
    def assess_separation_quality(self, separated_audio: np.ndarray, 
                                   original_audio: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Assess the quality of source separation.
        
        Args:
            separated_audio: Separated audio source
            original_audio: Original full mix (optional, for comparison)
        
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Signal-to-noise ratio (SNR) estimation
        metrics['snr'] = self._estimate_snr(separated_audio)
        
        # RMS energy
        metrics['rms'] = np.sqrt(np.mean(separated_audio**2))
        
        # Zero crossing rate
        metrics['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(separated_audio)[0])
        
        # Spectral centroid
        if len(separated_audio) > 0:
            stft = librosa.stft(separated_audio)
            spectral_centroid = librosa.feature.spectral_centroid(S=np.abs(stft))[0]
            metrics['spectral_centroid'] = np.mean(spectral_centroid)
        else:
            metrics['spectral_centroid'] = 0.0
        
        # Artifact detection
        metrics['artifacts_detected'] = self._detect_artifacts(separated_audio)
        
        # Overall confidence score
        metrics['confidence'] = self._calculate_confidence(metrics)
        
        return metrics
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        if len(audio) == 0:
            return 0.0
        
        # Simple SNR estimation based on signal power vs noise floor
        signal_power = np.mean(audio**2)
        noise_floor = np.percentile(np.abs(audio), 10)**2  # Estimate noise from low percentile
        
        if noise_floor > 0:
            snr_db = 10 * np.log10(signal_power / noise_floor)
            return max(0.0, snr_db)
        return 0.0
    
    def _detect_artifacts(self, audio: np.ndarray) -> bool:
        """Detect artifacts in audio (clipping, distortion, etc.)."""
        if len(audio) == 0:
            return False
        
        # Check for clipping
        max_val = np.max(np.abs(audio))
        clipping = max_val > 0.95
        
        # Check for sudden discontinuities (potential artifacts)
        diff = np.diff(audio)
        large_jumps = np.sum(np.abs(diff) > 0.5) / len(diff) > 0.01  # More than 1% large jumps
        
        return clipping or large_jumps
    
    def _calculate_confidence(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall confidence score from metrics."""
        confidence = 0.0
        
        # SNR contribution (0-0.4)
        if metrics['snr'] is not None:
            snr_score = min(metrics['snr'] / 30.0, 1.0)  # Normalize to 30dB max
            confidence += snr_score * 0.4
        
        # RMS contribution (0-0.3)
        if metrics['rms'] is not None:
            rms_score = min(metrics['rms'] * 10, 1.0)
            confidence += rms_score * 0.3
        
        # Artifact penalty (0-0.3)
        if not metrics['artifacts_detected']:
            confidence += 0.3
        
        return min(confidence, 1.0)
    
    def compare_separations(self, separation1: Dict[str, Any], 
                           separation2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two separation results."""
        return {
            'separation1_better': separation1['confidence'] > separation2['confidence'],
            'confidence_diff': separation1['confidence'] - separation2['confidence'],
            'snr_diff': separation1.get('snr', 0) - separation2.get('snr', 0)
        }

