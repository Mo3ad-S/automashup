"""
Main mashup engine that orchestrates the entire enhanced pipeline.
"""
import numpy as np
import copy
from typing import List, Dict, Any, Optional
from automashup.src.core.track import Track
from automashup.src.alignment.beat_sync import BeatSynchronizer
from automashup.src.alignment.tempo_matcher import TempoMatcher
from automashup.src.alignment.pitch_corrector import PitchCorrector
from automashup.src.inpainting.silence_handler import SilenceHandler
from automashup.src.inpainting.audio_inpainter import AudioInpainter
from automashup.src.enhancement.enhancer import AudioEnhancer
from automashup.src.enhancement.vocal_enhancer import VocalEnhancer
from automashup.src.enhancement.instrumental_enhancer import InstrumentalEnhancer
from automashup.src.mixing.stem_mixer import StemMixer
from automashup.src.mixing.transition_smoother import TransitionSmoother
from automashup.src.postprocessing.mastering import Mastering
from automashup.src.postprocessing.quality_checker import QualityChecker


class MashupEngine:
    """Main mashup engine orchestrating the enhanced pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize mashup engine.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._default_config()
        
        # Initialize modules
        self.beat_synchronizer = BeatSynchronizer(
            crossfade_duration=self.config.get('crossfade_duration', 0.1)
        )
        self.tempo_matcher = TempoMatcher(
            use_phase_vocoder=self.config.get('use_phase_vocoder', True)
        )
        self.pitch_corrector = PitchCorrector(
            preserve_formants=self.config.get('preserve_formants', True)
        )
        self.silence_handler = SilenceHandler(
            sr=self.config.get('sr', 44100)
        )
        self.audio_inpainter = AudioInpainter(
            use_gpu=self.config.get('use_gpu', True),
            model_name=self.config.get('inpainting_model', 'interpolation')
        )
        self.enhancer = AudioEnhancer(
            use_gpu=self.config.get('use_gpu', True)
        )
        self.vocal_enhancer = VocalEnhancer(
            sr=self.config.get('sr', 44100)
        )
        self.instrumental_enhancer = InstrumentalEnhancer(
            sr=self.config.get('sr', 44100)
        )
        self.stem_mixer = StemMixer(
            sr=self.config.get('sr', 44100)
        )
        self.transition_smoother = TransitionSmoother(
            sr=self.config.get('sr', 44100)
        )
        self.mastering = Mastering(
            sr=self.config.get('sr', 44100),
            target_lufs=self.config.get('target_lufs', -14.0)
        )
        self.quality_checker = QualityChecker(
            sr=self.config.get('sr', 44100)
        )
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'sr': 44100,
            'use_gpu': True,
            'use_phase_vocoder': True,
            'preserve_formants': True,
            'crossfade_duration': 0.1,
            'inpainting_model': 'interpolation',
            'target_lufs': -14.0,
            'enable_enhancement': True,
            'enable_inpainting': True,
            'enable_mixing': True,
            'enable_mastering': True,
            'quality_threshold': 0.7
        }
    
    def create_mashup(self, tracks: List[Track]) -> Track:
        """
        Create mashup from tracks using enhanced pipeline.
        
        Args:
            tracks: List of Track objects (first is reference/vocals, rest are instrumental)
        
        Returns:
            Final mashup Track
        """
        if not tracks:
            raise ValueError("No tracks provided")
        
        # Use first track as reference
        reference_track = tracks[0]
        other_tracks = tracks[1:]
        
        # Validate track sizes to prevent memory issues
        max_samples = self.config.get('max_track_samples', 50_000_000)
        safe_mode = self.config.get('safe_mode', True)
        
        for i, track in enumerate(tracks):
            if len(track.audio) > max_samples:
                print(f"Warning: Track {i} is very long ({len(track.audio)} samples). Truncating to {max_samples} samples.")
                track.audio = track.audio[:max_samples]
        
        print("Starting enhanced mashup pipeline...")
        
        # Step 1: Alignment & Synchronization
        print("Step 1: Aligning and synchronizing tracks...")
        if safe_mode:
            # Use simple alignment in safe mode
            print("  Using safe mode (simple alignment)...")
            aligned_tracks = self._simple_align_tracks(reference_track, other_tracks)
        else:
            try:
                aligned_tracks = self._align_tracks(reference_track, other_tracks)
            except (MemoryError, Exception) as e:
                print(f"Memory error during alignment: {e}")
                print("Falling back to simple alignment...")
                aligned_tracks = self._simple_align_tracks(reference_track, other_tracks)
        
        # Step 2: Tempo Matching
        print("Step 2: Matching tempo...")
        tempo_matched_tracks = self._match_tempo(aligned_tracks, reference_track)
        
        # Step 3: Pitch Correction
        print("Step 3: Correcting pitch...")
        pitch_corrected_tracks = self._correct_pitch(tempo_matched_tracks, reference_track)
        
        # Step 4: Silence Detection & Inpainting
        enable_inpainting = self.config.get('enable_inpainting', 
                                           self.config.get('inpainting_enabled', True))
        if enable_inpainting and not safe_mode:
            print("Step 4: Detecting and inpainting silences...")
            try:
                inpainted_tracks = self._inpaint_silences(pitch_corrected_tracks)
            except (MemoryError, Exception) as e:
                print(f"  Memory error during inpainting: {e}, skipping...")
                inpainted_tracks = pitch_corrected_tracks
        else:
            inpainted_tracks = pitch_corrected_tracks
        
        # Step 5: Quality Enhancement
        enable_enhancement = self.config.get('enable_enhancement',
                                            self.config.get('enhancement_enabled', True))
        if enable_enhancement and not safe_mode:
            print("Step 5: Enhancing quality...")
            try:
                enhanced_tracks = self._enhance_quality(inpainted_tracks)
            except (MemoryError, Exception) as e:
                print(f"  Memory error during enhancement: {e}, skipping...")
                enhanced_tracks = inpainted_tracks
        else:
            enhanced_tracks = inpainted_tracks
        
        # Step 6: Mixing
        enable_mixing = self.config.get('enable_mixing',
                                       self.config.get('mixing_enabled', True))
        if enable_mixing and not safe_mode:
            print("Step 6: Mixing tracks...")
            try:
                mixed_audio = self._mix_tracks(enhanced_tracks)
            except (MemoryError, Exception) as e:
                print(f"  Memory error during mixing: {e}, using simple mix...")
                mixed_audio = self._simple_mix(enhanced_tracks)
        else:
            print("Step 6: Mixing tracks (simple mode)...")
            mixed_audio = self._simple_mix(enhanced_tracks)
        
        # Step 7: Transition Smoothing
        print("Step 7: Smoothing transitions...")
        smoothed_audio = self._smooth_transitions(mixed_audio, reference_track)
        
        # Step 8: Mastering
        enable_mastering = self.config.get('enable_mastering',
                                          self.config.get('mastering_enabled', True))
        if enable_mastering:
            print("Step 8: Mastering...")
            mastered_audio = self._master(smoothed_audio)
        else:
            mastered_audio = smoothed_audio
        
        # Step 9: Quality Check
        print("Step 9: Checking quality...")
        quality_metrics = self.quality_checker.check_quality(mastered_audio)
        
        # Create result track
        result_track = Track(
            track_name=f"Mashup - {reference_track.name}",
            audio=mastered_audio,
            metadata=reference_track.__dict__.copy(),
            sr=reference_track.sr
        )
        result_track.quality_metrics = quality_metrics
        
        # Reprocess if quality is low
        if self.quality_checker.should_reprocess(quality_metrics, 
                                                self.config.get('quality_threshold', 0.7)):
            print("Quality below threshold, reprocessing...")
            # Could implement reprocessing logic here
        
        print(f"Mashup complete! Quality score: {quality_metrics.get('quality_score', 0.0):.2f}")
        
        return result_track
    
    def _align_tracks(self, reference_track: Track, other_tracks: List[Track]) -> List[Track]:
        """Align tracks beat-synchronously."""
        aligned = [reference_track]
        
        for i, track in enumerate(other_tracks):
            try:
                print(f"  Aligning track {i+1}/{len(other_tracks)}...")
                aligned_track = self.beat_synchronizer.align_tracks_beat_sync(
                    copy.deepcopy(track), reference_track
                )
                aligned.append(aligned_track)
            except MemoryError as e:
                print(f"  Memory error aligning track {i+1}: {e}")
                print(f"  Using simple alignment for this track...")
                # Use simple fallback
                aligned_track = self._simple_align_track(track, reference_track)
                aligned.append(aligned_track)
        
        return aligned
    
    def _simple_align_tracks(self, reference_track: Track, other_tracks: List[Track]) -> List[Track]:
        """Simple alignment fallback that doesn't use beat-sync."""
        aligned = [reference_track]
        
        for track in other_tracks:
            aligned_track = self._simple_align_track(track, reference_track)
            aligned.append(aligned_track)
        
        return aligned
    
    def _simple_align_track(self, track: Track, reference_track: Track) -> Track:
        """Simple alignment - just match length and tempo."""
        import copy as cp
        aligned = cp.deepcopy(track)
        
        # Simple approach: just match length
        target_length = len(reference_track.audio)
        current_length = len(aligned.audio)
        
        if current_length < target_length:
            # Pad with silence
            padding = target_length - current_length
            aligned.audio = np.pad(aligned.audio, (0, padding), mode='constant')
        elif current_length > target_length:
            # Truncate
            aligned.audio = aligned.audio[:target_length]
        
        # Update beats proportionally
        if hasattr(reference_track, 'beats') and reference_track.beats:
            ratio = len(aligned.audio) / current_length if current_length > 0 else 1.0
            aligned.beats = [b * ratio for b in track.beats[:len(reference_track.beats)]]
            aligned.downbeats = [db * ratio for db in track.downbeats[:len(reference_track.downbeats)]]
        
        return aligned
    
    def _match_tempo(self, tracks: List[Track], reference_track: Track) -> List[Track]:
        """Match tempo of all tracks to reference."""
        target_bpm = reference_track.bpm if hasattr(reference_track, 'bpm') and reference_track.bpm else 120
        
        matched = []
        for track in tracks:
            if track == reference_track:
                matched.append(track)
            else:
                matched_track = self.tempo_matcher.match_tempo_multi_stage(copy.deepcopy(track), target_bpm)
                matched.append(matched_track)
        
        return matched
    
    def _correct_pitch(self, tracks: List[Track], reference_track: Track) -> List[Track]:
        """Correct pitch of all tracks to reference."""
        target_key = reference_track.get_key()
        if target_key is None:
            return tracks
        
        corrected = []
        for track in tracks:
            if track == reference_track:
                corrected.append(track)
            else:
                corrected_track = self.pitch_corrector.correct_pitch(copy.deepcopy(track), target_key)
                corrected.append(corrected_track)
        
        return corrected
    
    def _inpaint_silences(self, tracks: List[Track]) -> List[Track]:
        """Inpaint silences in tracks."""
        inpainted = []
        
        for track in tracks:
            # Detect silences
            silence_regions = self.silence_handler.prepare_inpainting_regions(track)
            
            # Inpaint
            if silence_regions:
                inpainted_audio = self.audio_inpainter.inpaint(
                    track.audio,
                    silence_regions,
                    track.sr
                )
                track.audio = inpainted_audio
            
            inpainted.append(track)
        
        return inpainted
    
    def _enhance_quality(self, tracks: List[Track]) -> List[Track]:
        """Enhance quality of tracks."""
        enhanced = []
        
        for track in tracks:
            # General enhancement
            enhanced_audio = self.enhancer.enhance(
                track.audio,
                track.sr
            )
            
            # Track-specific enhancement
            if track.track_type == 'vocals':
                enhanced_audio = self.vocal_enhancer.enhance_vocals(
                    enhanced_audio,
                    track.sr
                )
            elif track.track_type in ['instrumental', 'bass', 'drums', 'other']:
                enhanced_audio = self.instrumental_enhancer.enhance_instrumental(
                    enhanced_audio,
                    track.sr
                )
            
            track.audio = enhanced_audio
            enhanced.append(track)
        
        return enhanced
    
    def _mix_tracks(self, tracks: List[Track]) -> np.ndarray:
        """Mix tracks using stem mixer."""
        return self.stem_mixer.mix_stems(tracks)
    
    def _simple_mix(self, tracks: List[Track]) -> np.ndarray:
        """Simple mixing (fallback)."""
        if not tracks:
            return np.array([])
        
        # Find max length
        max_length = max(len(track.audio) for track in tracks)
        
        # Sum all tracks
        mixed = np.zeros(max_length)
        for track in tracks:
            padded = np.pad(track.audio, (0, max_length - len(track.audio)), mode='constant')
            mixed += padded
        
        # Normalize
        max_val = np.max(np.abs(mixed))
        if max_val > 0.95:
            mixed = mixed * 0.95 / max_val
        
        return mixed
    
    def _smooth_transitions(self, audio: np.ndarray, reference_track: Track) -> np.ndarray:
        """Smooth transitions in audio."""
        # Get transition points from reference track segments
        transition_points = []
        for segment in reference_track.segments:
            transition_points.append(segment.start)
            transition_points.append(segment.end)
        
        # Remove duplicates and sort
        transition_points = sorted(set(transition_points))
        
        # Smooth transitions
        smoothed = self.transition_smoother.smooth_transitions(
            audio,
            transition_points
        )
        
        return smoothed
    
    def _master(self, audio: np.ndarray) -> np.ndarray:
        """Master audio."""
        return self.mastering.master(audio)

