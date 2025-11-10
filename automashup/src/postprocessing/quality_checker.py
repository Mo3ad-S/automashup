"""
Quality checker module for artifact detection and quality metrics.
"""
import numpy as np
import librosa
from typing import Dict, Any, List, Tuple


class QualityChecker:
    """Quality checker for final output."""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
    
    def check_quality(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Check quality of audio.
        
        Args:
            audio: Input audio
        
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Artifact detection
        metrics['artifacts_detected'] = self.detect_artifacts(audio)
        
        # Silence detection
        metrics['silences'] = self.detect_silences(audio)
        
        # Audio duration
        metrics['audio_duration'] = len(audio) / self.sr if len(audio) > 0 else 1.0
        
        # Quality metrics
        metrics['snr'] = self.calculate_snr(audio)
        metrics['rms'] = self.calculate_rms(audio)
        metrics['spectral_centroid'] = self.calculate_spectral_centroid(audio)
        
        # Overall quality score
        metrics['quality_score'] = self.calculate_quality_score(metrics)
        
        return metrics
    
    def detect_artifacts(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Detect artifacts in audio.
        
        Args:
            audio: Input audio
        
        Returns:
            Dictionary with artifact information
        """
        artifacts = {
            'clipping': False,
            'discontinuities': False,
            'phase_artifacts': False,
            'spectral_artifacts': False
        }
        
        if len(audio) == 0:
            return artifacts
        
        # Check for clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0.99:
            artifacts['clipping'] = True
        
        # Check for discontinuities
        diff = np.diff(audio)
        threshold = np.std(diff) * 3
        large_jumps = np.sum(np.abs(diff) > threshold) / len(diff)
        if large_jumps > 0.01:  # More than 1% large jumps
            artifacts['discontinuities'] = True
        
        # Check for phase artifacts (simplified)
        stft = librosa.stft(audio)
        phase = np.angle(stft)
        phase_diff = np.diff(phase, axis=1)
        phase_jumps = np.sum(np.abs(phase_diff) > np.pi) / phase_diff.size
        if phase_jumps > 0.1:  # More than 10% phase jumps
            artifacts['phase_artifacts'] = True
        
        # Check for spectral artifacts
        magnitude = np.abs(stft)
        magnitude_mean = np.mean(magnitude, axis=1)
        magnitude_std = np.std(magnitude, axis=1)
        outliers = np.sum(magnitude_std > 2 * magnitude_mean) / len(magnitude_mean)
        if outliers > 0.05:  # More than 5% outliers
            artifacts['spectral_artifacts'] = True
        
        return artifacts
    
    def detect_silences(self, audio: np.ndarray, 
                       threshold_db: float = -40.0,
                       min_duration: float = 0.1) -> List[Dict[str, Any]]:
        """
        Detect silences in audio.
        
        Args:
            audio: Input audio
            threshold_db: Threshold in dB
            min_duration: Minimum duration in seconds
        
        Returns:
            List of silence regions
        """
        if len(audio) == 0:
            return []
        
        # Calculate RMS per frame
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, 
                                  hop_length=hop_length)[0]
        
        # Convert to dB
        rms_db = librosa.power_to_db(rms**2, ref=np.max(rms**2))
        
        # Find frames below threshold
        silence_frames = rms_db < threshold_db
        
        # Convert to time
        times = librosa.frames_to_time(np.arange(len(silence_frames)), 
                                      sr=self.sr, hop_length=hop_length)
        
        # Find continuous silence regions
        silences = []
        in_silence = False
        silence_start = 0.0
        
        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_silence:
                in_silence = True
                silence_start = times[i]
            elif not is_silent and in_silence:
                in_silence = False
                silence_end = times[i]
                duration = silence_end - silence_start
                if duration >= min_duration:
                    silences.append({
                        'start': silence_start,
                        'end': silence_end,
                        'duration': duration
                    })
        
        # Handle silence at end
        if in_silence:
            silence_end = times[-1]
            duration = silence_end - silence_start
            if duration >= min_duration:
                silences.append({
                    'start': silence_start,
                    'end': silence_end,
                    'duration': duration
                })
        
        return silences
    
    def calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        if len(audio) == 0:
            return 0.0
        
        signal_power = np.mean(audio**2)
        noise_floor = np.percentile(np.abs(audio), 10)**2
        
        if noise_floor > 0:
            snr_db = 10 * np.log10(signal_power / noise_floor)
            return max(0.0, snr_db)
        return 0.0
    
    def calculate_rms(self, audio: np.ndarray) -> float:
        """Calculate RMS energy."""
        if len(audio) == 0:
            return 0.0
        return np.sqrt(np.mean(audio**2))
    
    def calculate_spectral_centroid(self, audio: np.ndarray) -> float:
        """Calculate spectral centroid."""
        if len(audio) == 0:
            return 0.0
        
        stft = librosa.stft(audio)
        spectral_centroid = librosa.feature.spectral_centroid(S=np.abs(stft))[0]
        return np.mean(spectral_centroid)
    
    def calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate overall quality score.
        
        Args:
            metrics: Quality metrics dictionary
        
        Returns:
            Quality score (0-1)
        """
        score = 0.0
        
        # SNR contribution (0-0.4)
        if metrics.get('snr') is not None:
            snr_score = min(metrics['snr'] / 30.0, 1.0)
            score += snr_score * 0.4
        
        # RMS contribution (0-0.2)
        if metrics.get('rms') is not None:
            rms_score = min(metrics['rms'] * 10, 1.0)
            score += rms_score * 0.2
        
        # Artifact penalty (0-0.2)
        artifacts = metrics.get('artifacts_detected', {})
        artifact_count = sum(artifacts.values())
        if artifact_count == 0:
            score += 0.2
        else:
            score += max(0, 0.2 - artifact_count * 0.05)
        
        # Silence penalty (0-0.2)
        silences = metrics.get('silences', [])
        if len(silences) == 0:
            score += 0.2
        else:
            total_silence = sum(s['duration'] for s in silences)
            audio_duration = metrics.get('audio_duration', 1.0)
            silence_ratio = total_silence / audio_duration
            score += max(0, 0.2 - silence_ratio * 0.2)
        
        return min(score, 1.0)
    
    def should_reprocess(self, metrics: Dict[str, Any], 
                        threshold: float = 0.7) -> bool:
        """
        Determine if audio should be reprocessed.
        
        Args:
            metrics: Quality metrics
            threshold: Quality threshold
        
        Returns:
            True if should reprocess
        """
        quality_score = metrics.get('quality_score', 0.0)
        return quality_score < threshold

