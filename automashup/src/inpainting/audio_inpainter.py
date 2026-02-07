"""
Audio inpainting module with generative models and fallback methods.

Supports multiple backends:
- 'interpolation': Fast, lightweight crossfade/interpolation (default)
- 'stable_audio': High-quality generative inpainting using Stable Audio Open 1.0
"""
import numpy as np
import torch
import torchaudio
from typing import Dict, Any, List, Optional, Tuple
import librosa


class AudioInpainter:
    """Audio inpainting with generative models and fallback methods."""
    
    def __init__(self, use_gpu: bool = True, model_name: str = 'interpolation'):
        """
        Initialize audio inpainter.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            model_name: Backend to use - 'interpolation' (fast), 'stable_audio' (high quality)
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.model_name = model_name
        self.model = None
        self._stable_audio_inpainter = None
        
        # Load model if needed
        if model_name != 'interpolation':
            self._load_model()
    
    def _load_model(self):
        """Load inpainting model based on model_name."""
        if self.model_name == 'stable_audio':
            try:
                from automashup.src.inpainting.stable_audio_inpainter import StableAudioInpainter
                self._stable_audio_inpainter = StableAudioInpainter(use_gpu=self.use_gpu)
                print("Stable Audio inpainter initialized (model will load on first use)")
            except ImportError as e:
                print(f"Could not load Stable Audio: {e}")
                print("Falling back to interpolation")
                self.model_name = 'interpolation'
        # Add other model backends here as needed
        pass
    
    def inpaint(self, audio: np.ndarray, silence_regions: List[Dict[str, Any]], 
                sr: int = 44100) -> np.ndarray:
        """
        Inpaint silence regions in audio.
        
        Args:
            audio: Input audio with silences
            silence_regions: List of silence region dictionaries
            sr: Sample rate
        
        Returns:
            Audio with silences filled
        """
        result_audio = audio.copy()
        
        # Process each silence region
        for region in sorted(silence_regions, key=lambda x: x['start'], reverse=True):
            # Process from end to start to avoid index shifting
            start_idx = region['start_samples']
            end_idx = region['end_samples']
            duration = region['duration']
            silence_type = region['type']
            
            # Choose inpainting method based on duration
            if duration < 0.5:
                # Short gap: use crossfade/interpolation
                filled = self._inpaint_short_gap(
                    result_audio,
                    start_idx,
                    end_idx,
                    region.get('context', {})
                )
            elif duration < 2.0:
                # Medium gap: use generative inpainting or interpolation
                filled = self._inpaint_medium_gap(
                    result_audio,
                    start_idx,
                    end_idx,
                    region.get('context', {}),
                    sr=sr
                )
            else:
                # Long gap: use segment repetition or generative model
                filled = self._inpaint_long_gap(
                    result_audio,
                    start_idx,
                    end_idx,
                    region.get('context', {}),
                    sr=sr
                )
            
            # Replace silence with filled audio
            result_audio = np.concatenate([
                result_audio[:start_idx],
                filled,
                result_audio[end_idx:]
            ])
        
        return result_audio
    
    def _inpaint_short_gap(self, audio: np.ndarray, start_idx: int, 
                          end_idx: int, context: Dict[str, Any]) -> np.ndarray:
        """Inpaint short gaps using crossfade/interpolation."""
        gap_length = end_idx - start_idx
        
        # Get context before and after
        context_before = context.get('context_before', np.array([]))
        context_after = context.get('context_after', np.array([]))
        
        if len(context_before) > 0 and len(context_after) > 0:
            # Crossfade between before and after
            before_tail = context_before[-gap_length:] if len(context_before) >= gap_length else context_before
            after_head = context_after[:gap_length] if len(context_after) >= gap_length else context_after
            
            # Linear interpolation
            if len(before_tail) == gap_length and len(after_head) == gap_length:
                fade_out = np.linspace(1.0, 0.0, gap_length)
                fade_in = np.linspace(0.0, 1.0, gap_length)
                filled = before_tail * fade_out + after_head * fade_in
            else:
                # Simple interpolation
                filled = np.linspace(
                    before_tail[-1] if len(before_tail) > 0 else 0.0,
                    after_head[0] if len(after_head) > 0 else 0.0,
                    gap_length
                )
        elif len(context_before) > 0:
            # Extend from before
            if len(context_before) >= gap_length:
                filled = context_before[-gap_length:]
            else:
                filled = np.tile(context_before, (gap_length // len(context_before)) + 1)[:gap_length]
        elif len(context_after) > 0:
            # Extend from after
            if len(context_after) >= gap_length:
                filled = context_after[:gap_length]
            else:
                filled = np.tile(context_after, (gap_length // len(context_after)) + 1)[:gap_length]
        else:
            # No context: use zero padding with fade
            filled = np.zeros(gap_length)
        
        return filled
    
    def _inpaint_medium_gap(self, audio: np.ndarray, start_idx: int, 
                           end_idx: int, context: Dict[str, Any],
                           sr: int = 44100) -> np.ndarray:
        """Inpaint medium gaps using generative model or interpolation."""
        gap_length = end_idx - start_idx
        
        # Try generative model if available (Stable Audio or other)
        if self._stable_audio_inpainter is not None or self.model is not None:
            try:
                filled = self._inpaint_with_model(audio, start_idx, end_idx, context, sr=sr)
                if filled is not None:
                    return filled
            except Exception as e:
                print(f"Model inpainting failed: {e}, using fallback")
        
        # Fallback to intelligent interpolation
        return self._inpaint_intelligent_interpolation(audio, start_idx, end_idx, context)
    
    def _inpaint_long_gap(self, audio: np.ndarray, start_idx: int, 
                         end_idx: int, context: Dict[str, Any],
                         sr: int = 44100) -> np.ndarray:
        """Inpaint long gaps using segment repetition or generative model."""
        gap_length = end_idx - start_idx
        
        # Try generative model if available (Stable Audio or other)
        if self._stable_audio_inpainter is not None or self.model is not None:
            try:
                filled = self._inpaint_with_model(audio, start_idx, end_idx, context, sr=sr)
                if filled is not None:
                    return filled
            except Exception as e:
                print(f"Model inpainting failed: {e}, using fallback")
        
        # Fallback to segment repetition with variation
        return self._inpaint_segment_repetition(audio, start_idx, end_idx, context)
    
    def _inpaint_with_model(self, audio: np.ndarray, start_idx: int, 
                           end_idx: int, context: Dict[str, Any],
                           sr: int = 44100) -> Optional[np.ndarray]:
        """Inpaint using generative model."""
        if self.model_name == 'stable_audio' and self._stable_audio_inpainter is not None:
            try:
                # Use Stable Audio for inpainting
                gap_length = end_idx - start_idx
                
                # Generate audio for this region
                inpainted_audio = self._stable_audio_inpainter.inpaint_region(
                    audio, sr, start_idx, end_idx,
                    prompt=None,  # Auto-generate prompt from context
                    crossfade_duration=0.1
                )
                
                # Extract just the inpainted region
                return inpainted_audio[start_idx:end_idx]
            except Exception as e:
                print(f"Stable Audio inpainting failed: {e}")
                return None
        
        # Fallback: return None to use interpolation
        return None
    
    def _inpaint_intelligent_interpolation(self, audio: np.ndarray, 
                                          start_idx: int, end_idx: int,
                                          context: Dict[str, Any]) -> np.ndarray:
        """Intelligent interpolation using spectral features."""
        gap_length = end_idx - start_idx
        
        context_before = context.get('context_before', np.array([]))
        context_after = context.get('context_after', np.array([]))
        
        if len(context_before) > 0 and len(context_after) > 0:
            # Use spectral features for better interpolation
            # Extract spectral features
            features_before = context.get('spectral_features_before', {})
            features_after = context.get('spectral_features_after', {})
            
            # Interpolate in spectral domain
            if features_before and features_after:
                # Use chroma-based interpolation
                chroma_before = features_before.get('chroma_mean', np.zeros(12))
                chroma_after = features_after.get('chroma_mean', np.zeros(12))
                
                # Create transition
                n_frames = gap_length // 512  # Approximate frame count
                if n_frames > 0:
                    # Interpolate chroma
                    chroma_interp = np.linspace(chroma_before, chroma_after, n_frames).T
                    
                    # Convert back to audio (simplified)
                    # In full implementation, would use inverse chroma synthesis
                    filled = self._inpaint_short_gap(audio, start_idx, end_idx, context)
                else:
                    filled = self._inpaint_short_gap(audio, start_idx, end_idx, context)
            else:
                filled = self._inpaint_short_gap(audio, start_idx, end_idx, context)
        else:
            filled = self._inpaint_short_gap(audio, start_idx, end_idx, context)
        
        return filled
    
    def _inpaint_segment_repetition(self, audio: np.ndarray, 
                                    start_idx: int, end_idx: int,
                                    context: Dict[str, Any]) -> np.ndarray:
        """Inpaint using segment repetition with variation."""
        gap_length = end_idx - start_idx
        
        context_before = context.get('context_before', np.array([]))
        context_after = context.get('context_after', np.array([]))
        
        # Use longer context for repetition
        if len(context_before) > 0:
            # Repeat from before with variation
            segment_length = min(len(context_before), gap_length)
            if segment_length > 0:
                # Repeat segment
                n_repeats = (gap_length // segment_length) + 1
                repeated = np.tile(context_before[-segment_length:], n_repeats)[:gap_length]
                
                # Add slight variation (pitch/time variation)
                # Simplified: just add small random variation
                variation = np.random.normal(0, 0.01, gap_length)
                filled = repeated + variation
            else:
                filled = np.zeros(gap_length)
        elif len(context_after) > 0:
            # Repeat from after
            segment_length = min(len(context_after), gap_length)
            if segment_length > 0:
                n_repeats = (gap_length // segment_length) + 1
                repeated = np.tile(context_after[:segment_length], n_repeats)[:gap_length]
                variation = np.random.normal(0, 0.01, gap_length)
                filled = repeated + variation
            else:
                filled = np.zeros(gap_length)
        else:
            # No context: use zero padding
            filled = np.zeros(gap_length)
        
        return filled

