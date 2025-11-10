"""
Beat-synchronous alignment module with intelligent crossfading.
"""
import numpy as np
import librosa
from typing import List, Tuple, Optional
from automashup.src.core.track import Track
from automashup.src.core.segment import Segment


class BeatSynchronizer:
    """Beat-synchronous alignment with crossfading."""
    
    def __init__(self, crossfade_duration: float = 0.1):
        self.crossfade_duration = crossfade_duration
    
    def align_tracks_beat_sync(self, source_track: Track, target_track: Track) -> Track:
        """
        Align source track to target track using beat-synchronous alignment.
        
        Args:
            source_track: Track to align
            target_track: Reference track
        
        Returns:
            Aligned source track
        """
        # Validate input sizes
        max_samples = 50_000_000  # 50M samples max
        if len(source_track.audio) > max_samples:
            print(f"Warning: Source track too long ({len(source_track.audio)} samples), truncating...")
            source_track.audio = source_track.audio[:max_samples]
        if len(target_track.audio) > max_samples:
            print(f"Warning: Target track too long ({len(target_track.audio)} samples), truncating...")
            target_track.audio = target_track.audio[:max_samples]
        
        aligned_audio = np.array([])
        aligned_beats = []
        aligned_downbeats = []
        
        # Limit number of segments to process to prevent memory issues
        max_segments = 50
        segments_to_process = target_track.segments[:max_segments] if len(target_track.segments) > max_segments else target_track.segments
        
        # Use target track's structure as reference
        for i, target_segment in enumerate(segments_to_process):
            # Find matching segment in source track
            matching_segment = self._find_matching_segment(
                source_track.segments, 
                target_segment.label
            )
            
            if matching_segment:
                # Align segment beat-synchronously
                aligned_seg, seg_beats, seg_downbeats = self._align_segment_beat_sync(
                    matching_segment,
                    target_segment
                )
            else:
                # Create silence or use fallback
                aligned_seg, seg_beats, seg_downbeats = self._create_fallback_segment(
                    target_segment,
                    source_track.sr
                )
            
            # Crossfade with previous segment
            if len(aligned_audio) > 0:
                try:
                    aligned_audio = self._crossfade_segments(
                        aligned_audio,
                        aligned_seg,
                        source_track.sr
                    )
                except MemoryError:
                    # If crossfade fails, just concatenate
                    print(f"  Memory error in crossfade for segment {i+1}, using simple concatenation")
                    aligned_audio = np.concatenate([aligned_audio, aligned_seg])
            else:
                aligned_audio = aligned_seg
            
            # Limit total length to prevent memory issues
            max_total_samples = 50_000_000
            if len(aligned_audio) > max_total_samples:
                print(f"  Warning: Total aligned audio exceeds {max_total_samples} samples, truncating")
                aligned_audio = aligned_audio[:max_total_samples]
                break
            
            # Update beats
            offset = len(aligned_audio) / source_track.sr
            aligned_beats.extend([b + offset for b in seg_beats])
            aligned_downbeats.extend([db + offset for db in seg_downbeats])
        
        # Ensure final audio is reasonable size
        max_final_samples = 50_000_000
        if len(aligned_audio) > max_final_samples:
            print(f"Warning: Final aligned audio too long, truncating to {max_final_samples} samples")
            aligned_audio = aligned_audio[:max_final_samples]
        
        # Update source track
        source_track.audio = aligned_audio
        source_track.beats = aligned_beats
        source_track.downbeats = aligned_downbeats
        
        return source_track
    
    def _find_matching_segment(self, segments: List[Segment], 
                              label: str) -> Optional[Segment]:
        """Find segment with matching label."""
        for segment in segments:
            if segment.label == label:
                return segment
        return None
    
    def _align_segment_beat_sync(self, source_seg: Segment, 
                                 target_seg: Segment) -> Tuple[np.ndarray, List[float], List[float]]:
        """Align segment beat-synchronously."""
        # Ensure 1D audio arrays - flatten completely
        source_audio = np.atleast_1d(source_seg.audio).flatten()
        target_audio = np.atleast_1d(target_seg.audio).flatten()
        
        # Validate array sizes
        max_size = 100_000_000  # 100M samples max
        if len(source_audio) > max_size or len(target_audio) > max_size:
            raise ValueError(f"Audio segment too large: {len(source_audio)}, {len(target_audio)}")
        
        # Calculate tempo ratio
        source_bpm = len(source_seg.beats) / source_seg.duration if source_seg.duration > 0 else 120
        target_bpm = len(target_seg.beats) / target_seg.duration if target_seg.duration > 0 else 120
        
        # Time stretch to match tempo (will be done by tempo_matcher)
        # For now, just prepare the segment
        target_duration_samples = len(target_audio)
        
        # Limit target duration to prevent huge arrays
        if target_duration_samples > max_size:
            target_duration_samples = max_size
            target_audio = target_audio[:max_size]
        
        # Stretch source segment to match target duration
        if len(source_audio) > 0:
            # Limit stretch ratio to prevent extreme values
            stretch_ratio = target_duration_samples / len(source_audio)
            if stretch_ratio > 5.0 or stretch_ratio < 0.2:
                # If ratio is too extreme, just pad or truncate
                if stretch_ratio > 5.0:
                    # Too much stretching needed - just pad
                    if target_duration_samples > len(source_audio):
                        aligned_audio = np.pad(source_audio, (0, target_duration_samples - len(source_audio)), mode='constant')
                    else:
                        aligned_audio = source_audio[:target_duration_samples]
                else:
                    # Too much compression needed - just truncate
                    aligned_audio = source_audio[:target_duration_samples]
            else:
                # Use librosa for better quality time stretching
                try:
                    aligned_audio = librosa.effects.time_stretch(
                        source_audio,
                        rate=1.0/stretch_ratio
                    )
                except Exception as e:
                    # Fallback to simple truncation/padding if time_stretch fails
                    print(f"Time stretch failed: {e}, using fallback")
                    if len(source_audio) < target_duration_samples:
                        aligned_audio = np.pad(source_audio, (0, target_duration_samples - len(source_audio)), mode='constant')
                    else:
                        aligned_audio = source_audio[:target_duration_samples]
            
            # Ensure correct length and 1D
            aligned_audio = np.atleast_1d(aligned_audio).flatten()
            if len(aligned_audio) > target_duration_samples:
                aligned_audio = aligned_audio[:target_duration_samples]
            elif len(aligned_audio) < target_duration_samples:
                aligned_audio = np.pad(
                    aligned_audio,
                    (0, target_duration_samples - len(aligned_audio)),
                    mode='constant'
                )
        else:
            aligned_audio = np.zeros(target_duration_samples)
        
        # Ensure 1D
        aligned_audio = np.atleast_1d(aligned_audio).flatten()
        
        # Map beats to target beats
        aligned_beats = target_seg.beats.copy() if hasattr(target_seg.beats, 'copy') else list(target_seg.beats)
        aligned_downbeats = target_seg.downbeats.copy() if hasattr(target_seg.downbeats, 'copy') else list(target_seg.downbeats)
        
        return aligned_audio, aligned_beats, aligned_downbeats
    
    def _create_fallback_segment(self, target_seg: Segment, 
                                 sr: int) -> Tuple[np.ndarray, List[float], List[float]]:
        """Create fallback segment (silence or repetition)."""
        target_duration_samples = len(target_seg.audio)
        
        # Create silence for now (will be filled by inpainting later)
        fallback_audio = np.zeros(target_duration_samples)
        
        return fallback_audio, target_seg.beats.copy(), target_seg.downbeats.copy()
    
    def _crossfade_segments(self, audio1: np.ndarray, audio2: np.ndarray, 
                           sr: int) -> np.ndarray:
        """Crossfade between two audio segments."""
        # Ensure 1D arrays - flatten completely
        audio1 = np.atleast_1d(audio1).flatten()
        audio2 = np.atleast_1d(audio2).flatten()
        
        # Validate array sizes to prevent memory issues
        max_size = 100_000_000  # 100M samples max (~2 hours at 44.1kHz)
        if len(audio1) > max_size or len(audio2) > max_size:
            raise ValueError(f"Audio array too large: {len(audio1)}, {len(audio2)}. Max allowed: {max_size}")
        
        crossfade_samples = int(self.crossfade_duration * sr)
        crossfade_samples = min(crossfade_samples, len(audio1), len(audio2))
        
        if crossfade_samples == 0:
            return np.concatenate([audio1, audio2])
        
        # Create fade curves
        fade_out = np.linspace(1.0, 0.0, crossfade_samples)
        fade_in = np.linspace(0.0, 1.0, crossfade_samples)
        
        # Apply crossfade - ensure all are 1D
        audio1_end = audio1[-crossfade_samples:].flatten()
        audio2_start = audio2[:crossfade_samples].flatten()
        
        # Ensure all arrays are same length for element-wise operations
        min_len = min(len(audio1_end), len(audio2_start), len(fade_out))
        if min_len != crossfade_samples:
            audio1_end = audio1_end[:min_len]
            audio2_start = audio2_start[:min_len]
            fade_out = fade_out[:min_len]
            fade_in = fade_in[:min_len]
            crossfade_samples = min_len
        
        # Element-wise multiplication (not broadcasting)
        crossfaded = audio1_end * fade_out + audio2_start * fade_in
        
        # Ensure result is 1D
        crossfaded = crossfaded.flatten()
        
        # Combine
        result = np.concatenate([
            audio1[:-crossfade_samples],
            crossfaded,
            audio2[crossfade_samples:]
        ])
        
        # Final validation
        if len(result.shape) > 1:
            result = result.flatten()
        
        return result
    
    def calculate_segment_similarity(self, seg1: Segment, seg2: Segment) -> float:
        """Calculate similarity between two segments."""
        if len(seg1.audio) == 0 or len(seg2.audio) == 0:
            return 0.0
        
        # Use chroma features for harmonic similarity
        chroma1 = librosa.feature.chroma_stft(y=seg1.audio, sr=seg1.sr)
        chroma2 = librosa.feature.chroma_stft(y=seg2.audio, sr=seg2.sr)
        
        # Normalize
        chroma1 = chroma1 / (np.linalg.norm(chroma1) + 1e-10)
        chroma2 = chroma2 / (np.linalg.norm(chroma2) + 1e-10)
        
        # Calculate cosine similarity
        similarity = np.mean(np.sum(chroma1 * chroma2, axis=0))
        
        return similarity

