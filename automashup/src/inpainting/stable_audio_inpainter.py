"""
Stable Audio-based inpainting for mashup improvement.

Uses Stable Audio Open 1.0 for high-quality audio generation to fill gaps,
smooth transitions, and improve overall mashup quality.

Fallback: If Stable Audio is not available, uses intelligent interpolation.
"""
import numpy as np
import torch
import torchaudio
from typing import Dict, Any, List, Optional, Tuple
import librosa

# Try to import einops, but don't fail if not available
try:
    from einops import rearrange
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False


class StableAudioInpainter:
    """Audio inpainting using Stable Audio diffusion model with fallback support."""
    
    def __init__(self, use_gpu: bool = True, model_id: str = "stabilityai/stable-audio-open-1.0",
                 fallback_mode: bool = True):
        """
        Initialize Stable Audio inpainter.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            model_id: HuggingFace model ID for Stable Audio
            fallback_mode: If True, use interpolation fallback when model unavailable
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        self.model_id = model_id
        self.model = None
        self.model_config = None
        self.sample_rate = 44100  # Default
        self.sample_size = 44100 * 47  # ~47 seconds default
        self._model_loaded = False
        self._model_available = None  # None = not checked yet
        self.fallback_mode = fallback_mode
        
    def _check_dependencies(self) -> Tuple[bool, str]:
        """Check if all dependencies are available."""
        errors = []
        
        # Check PyTorch version
        torch_version = torch.__version__.split('+')[0]
        major, minor = map(int, torch_version.split('.')[:2])
        if major < 2 or (major == 2 and minor < 1):
            errors.append(f"PyTorch 2.1+ required (found {torch_version}). "
                         "Upgrade with: pip install torch>=2.1")
        
        # Check stable-audio-tools
        try:
            import stable_audio_tools
        except ImportError:
            errors.append("stable-audio-tools not installed. "
                         "Install with: pip install stable-audio-tools")
        
        # Check einops
        if not EINOPS_AVAILABLE:
            errors.append("einops not installed. Install with: pip install einops")
        
        # Check k-diffusion compatibility
        try:
            import k_diffusion
        except ImportError:
            errors.append("k-diffusion not installed. Install with: pip install k-diffusion")
        
        if errors:
            return False, "\n".join(errors)
        return True, ""
        
    def load_model(self):
        """Load the Stable Audio model (lazy loading)."""
        if self._model_loaded:
            return True
        
        if self._model_available is False:
            # Already tried and failed
            return False
            
        # Check dependencies first
        deps_ok, error_msg = self._check_dependencies()
        if not deps_ok:
            print(f"⚠️  Stable Audio dependencies not satisfied:\n{error_msg}")
            if self.fallback_mode:
                print("ℹ️  Using fallback interpolation mode")
                self._model_available = False
                return False
            else:
                raise RuntimeError(error_msg)
        
        try:
            from stable_audio_tools import get_pretrained_model
            from stable_audio_tools.inference.generation import generate_diffusion_cond
            
            print(f"Loading Stable Audio model: {self.model_id}")
            self.model, self.model_config = get_pretrained_model(self.model_id)
            self.sample_rate = self.model_config["sample_rate"]
            self.sample_size = self.model_config["sample_size"]
            self.model = self.model.to(self.device)
            self._model_loaded = True
            self._model_available = True
            print(f"✓ Model loaded. Sample rate: {self.sample_rate}, Max samples: {self.sample_size}")
            return True
            
        except ImportError as e:
            error_msg = f"Import error: {e}\nInstall with: pip install stable-audio-tools einops"
            print(f"⚠️  {error_msg}")
            self._model_available = False
            if self.fallback_mode:
                print("ℹ️  Using fallback interpolation mode")
                return False
            raise ImportError(error_msg)
            
        except Exception as e:
            error_msg = f"Failed to load Stable Audio model: {e}"
            print(f"⚠️  {error_msg}")
            self._model_available = False
            if self.fallback_mode:
                print("ℹ️  Using fallback interpolation mode")
                return False
            raise RuntimeError(error_msg)
    
    def _generate_audio(self, prompt: str, duration_seconds: float = None,
                        negative_prompt: str = None, 
                        num_inference_steps: int = 100,
                        cfg_scale: float = 7.0,
                        context_audio: np.ndarray = None,
                        sr: int = 44100) -> np.ndarray:
        """
        Generate audio using Stable Audio or fallback interpolation.
        
        Args:
            prompt: Text description of desired audio
            duration_seconds: Duration of audio to generate (uses model max if None)
            negative_prompt: What to avoid in generation
            num_inference_steps: Number of diffusion steps
            cfg_scale: Classifier-free guidance scale
            context_audio: Optional context audio for fallback mode
            sr: Sample rate for fallback mode
            
        Returns:
            Generated audio as numpy array (mono)
        """
        # Try to load model, fall back if not available
        model_available = self.load_model()
        
        if not model_available:
            # Use fallback generation
            return self._generate_fallback(duration_seconds, context_audio, sr)
        
        from stable_audio_tools.inference.generation import generate_diffusion_cond
        
        # Build conditioning
        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": duration_seconds if duration_seconds else self.sample_size / self.sample_rate
        }]
        
        if negative_prompt:
            conditioning[0]["negative_prompt"] = negative_prompt
        
        # Generate audio
        with torch.no_grad():
            output = generate_diffusion_cond(
                self.model,
                conditioning=conditioning,
                sample_size=self.sample_size,
                device=self.device,
                steps=num_inference_steps,
                cfg_scale=cfg_scale
            )
        
        # Convert to numpy (mono)
        if EINOPS_AVAILABLE:
            output = rearrange(output, "b d n -> d (b n)")
        else:
            # Manual rearrange
            b, d, n = output.shape
            output = output.permute(1, 0, 2).reshape(d, b * n)
            
        output = output.to(torch.float32)
        
        # Normalize
        max_val = torch.max(torch.abs(output))
        if max_val > 0:
            output = output / max_val
        
        # Convert to mono numpy array
        output_np = output.mean(dim=0).cpu().numpy()
        
        return output_np
    
    def _generate_fallback(self, duration_seconds: float, 
                           context_audio: np.ndarray = None,
                           sr: int = 44100) -> np.ndarray:
        """
        Fallback audio generation using intelligent interpolation.
        
        Args:
            duration_seconds: Duration to generate
            context_audio: Context audio for interpolation
            sr: Sample rate
            
        Returns:
            Generated audio
        """
        target_samples = int(duration_seconds * sr) if duration_seconds else sr * 2
        
        if context_audio is not None and len(context_audio) > 0:
            # Use context audio with variation
            if len(context_audio) >= target_samples:
                # Use a portion of context
                generated = context_audio[:target_samples].copy()
            else:
                # Loop and blend context
                n_repeats = (target_samples // len(context_audio)) + 1
                generated = np.tile(context_audio, n_repeats)[:target_samples]
            
            # Add subtle variation to avoid exact repetition
            # Gentle amplitude modulation
            t = np.linspace(0, 2 * np.pi, target_samples)
            modulation = 1.0 + 0.05 * np.sin(t * 3)  # Very subtle
            generated = generated * modulation
            
            # Gentle fade in/out
            fade_len = min(int(0.05 * sr), target_samples // 4)
            if fade_len > 0:
                fade_in = np.linspace(0, 1, fade_len)
                fade_out = np.linspace(1, 0, fade_len)
                generated[:fade_len] *= fade_in
                generated[-fade_len:] *= fade_out
        else:
            # Generate silence with very low noise (better than pure silence)
            generated = np.random.normal(0, 0.001, target_samples).astype(np.float32)
        
        return generated
    
    def analyze_context(self, audio: np.ndarray, sr: int,
                        region_start: int, region_end: int,
                        context_duration: float = 2.0) -> Dict[str, Any]:
        """
        Analyze audio context around a region to generate appropriate prompt.
        
        Args:
            audio: Full audio array
            sr: Sample rate
            region_start: Start sample of region to inpaint
            region_end: End sample of region to inpaint
            context_duration: Duration of context to analyze (seconds)
            
        Returns:
            Dictionary with analysis results including generated prompt
        """
        context_samples = int(context_duration * sr)
        
        # Get context before and after
        before_start = max(0, region_start - context_samples)
        after_end = min(len(audio), region_end + context_samples)
        
        context_before = audio[before_start:region_start]
        context_after = audio[region_end:after_end]
        
        analysis = {
            'context_before': context_before,
            'context_after': context_after,
            'region_duration': (region_end - region_start) / sr,
        }
        
        # Analyze spectral characteristics
        if len(context_before) > sr // 4:  # Need at least 0.25s
            analysis['spectral_before'] = self._analyze_spectral(context_before, sr)
        if len(context_after) > sr // 4:
            analysis['spectral_after'] = self._analyze_spectral(context_after, sr)
        
        # Estimate tempo from context
        combined_context = np.concatenate([context_before, context_after])
        if len(combined_context) > sr:
            try:
                tempo, _ = librosa.beat.beat_track(y=combined_context, sr=sr)
                analysis['estimated_tempo'] = float(tempo)
            except:
                analysis['estimated_tempo'] = 120.0
        else:
            analysis['estimated_tempo'] = 120.0
        
        # Generate prompt based on analysis
        analysis['prompt'] = self._generate_prompt(analysis)
        
        return analysis
    
    def _analyze_spectral(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze spectral characteristics of audio."""
        try:
            # Spectral centroid (brightness)
            centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            avg_centroid = float(np.mean(centroid))
            
            # RMS energy
            rms = librosa.feature.rms(y=audio)
            avg_rms = float(np.mean(rms))
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            avg_rolloff = float(np.mean(rolloff))
            
            # Estimate if it's percussive or harmonic
            harmonic, percussive = librosa.effects.hpss(audio)
            harmonic_ratio = np.sum(np.abs(harmonic)) / (np.sum(np.abs(audio)) + 1e-8)
            
            return {
                'centroid': avg_centroid,
                'rms': avg_rms,
                'rolloff': avg_rolloff,
                'harmonic_ratio': float(harmonic_ratio),
                'is_percussive': harmonic_ratio < 0.5
            }
        except:
            return {}
    
    def _generate_prompt(self, analysis: Dict[str, Any]) -> str:
        """Generate a text prompt based on audio analysis."""
        parts = []
        
        # Add tempo
        tempo = analysis.get('estimated_tempo', 120)
        parts.append(f"{int(tempo)} BPM")
        
        # Analyze characteristics
        spectral_before = analysis.get('spectral_before', {})
        spectral_after = analysis.get('spectral_after', {})
        
        # Determine if percussive or melodic
        is_percussive = spectral_before.get('is_percussive', False) or spectral_after.get('is_percussive', False)
        
        if is_percussive:
            parts.append("drum pattern")
        else:
            parts.append("melodic instrumental")
        
        # Brightness
        centroid = spectral_before.get('centroid', 0) or spectral_after.get('centroid', 0)
        if centroid > 3000:
            parts.append("bright")
        elif centroid < 1500:
            parts.append("warm")
        
        # Energy level
        rms = spectral_before.get('rms', 0) or spectral_after.get('rms', 0)
        if rms > 0.1:
            parts.append("energetic")
        elif rms < 0.03:
            parts.append("soft ambient")
        
        # Add generic music descriptor
        parts.append("music loop")
        
        return ", ".join(parts)
    
    def inpaint_region(self, audio: np.ndarray, sr: int,
                       region_start: int, region_end: int,
                       prompt: str = None,
                       crossfade_duration: float = 0.1) -> np.ndarray:
        """
        Inpaint a specific region of audio.
        
        Args:
            audio: Full audio array
            sr: Sample rate
            region_start: Start sample of region to inpaint
            region_end: End sample of region to inpaint
            prompt: Optional custom prompt (auto-generated if None)
            crossfade_duration: Duration of crossfade in seconds
            
        Returns:
            Audio with region inpainted
        """
        # Try to load model (will use fallback if not available)
        model_available = self.load_model()
        
        region_duration = (region_end - region_start) / sr
        
        # Analyze context
        analysis = self.analyze_context(audio, sr, region_start, region_end)
        if prompt is None:
            prompt = analysis['prompt']
            print(f"Generated prompt: {prompt}")
        
        # Get context audio for fallback mode
        context_audio = np.concatenate([
            analysis.get('context_before', np.array([])),
            analysis.get('context_after', np.array([]))
        ]) if not model_available else None
        
        # Generate replacement audio
        # Note: Stable Audio has max duration, may need to generate in chunks
        max_duration = self.sample_size / self.sample_rate
        
        if region_duration <= max_duration:
            generated = self._generate_audio(
                prompt, duration_seconds=region_duration,
                context_audio=context_audio, sr=sr
            )
        else:
            # Generate in chunks and concatenate
            chunks = []
            remaining = region_duration
            while remaining > 0:
                chunk_duration = min(remaining, max_duration * 0.9)  # Slight overlap
                chunk = self._generate_audio(
                    prompt, duration_seconds=chunk_duration,
                    context_audio=context_audio, sr=sr
                )
                chunks.append(chunk)
                remaining -= chunk_duration
            generated = np.concatenate(chunks)
        
        # Resample if needed (only for model output, fallback already uses correct sr)
        if model_available and self.sample_rate != sr:
            generated = librosa.resample(generated, orig_sr=self.sample_rate, target_sr=sr)
        
        # Trim/pad to exact length
        target_length = region_end - region_start
        if len(generated) > target_length:
            generated = generated[:target_length]
        elif len(generated) < target_length:
            generated = np.pad(generated, (0, target_length - len(generated)))
        
        # Apply crossfade
        crossfade_samples = int(crossfade_duration * sr)
        crossfade_samples = min(crossfade_samples, target_length // 4)
        
        if crossfade_samples > 0:
            # Fade in from context before
            if region_start >= crossfade_samples:
                fade_in = np.linspace(0, 1, crossfade_samples)
                fade_out = np.linspace(1, 0, crossfade_samples)
                
                context_before = audio[region_start - crossfade_samples:region_start]
                generated[:crossfade_samples] = (
                    context_before * fade_out + 
                    generated[:crossfade_samples] * fade_in
                )
            
            # Fade out to context after
            if region_end + crossfade_samples <= len(audio):
                fade_in = np.linspace(0, 1, crossfade_samples)
                fade_out = np.linspace(1, 0, crossfade_samples)
                
                context_after = audio[region_end:region_end + crossfade_samples]
                generated[-crossfade_samples:] = (
                    generated[-crossfade_samples:] * fade_out + 
                    context_after * fade_in
                )
        
        # Replace region
        result = audio.copy()
        result[region_start:region_end] = generated
        
        return result
    
    def inpaint_silences(self, audio: np.ndarray, sr: int,
                         silence_threshold_db: float = -40.0,
                         min_silence_duration: float = 0.3,
                         max_silence_duration: float = 5.0) -> np.ndarray:
        """
        Detect and inpaint all silence regions in audio.
        
        Args:
            audio: Input audio
            sr: Sample rate
            silence_threshold_db: Threshold below which audio is considered silence
            min_silence_duration: Minimum silence duration to inpaint (seconds)
            max_silence_duration: Maximum silence duration to inpaint (seconds)
            
        Returns:
            Audio with silences inpainted
        """
        # Detect silences
        silence_regions = self._detect_silences(
            audio, sr, silence_threshold_db, 
            min_silence_duration, max_silence_duration
        )
        
        if not silence_regions:
            print("No significant silences detected")
            return audio
        
        print(f"Found {len(silence_regions)} silence regions to inpaint")
        
        # Process from end to start to avoid index shifting
        result = audio.copy()
        for i, (start, end) in enumerate(sorted(silence_regions, reverse=True)):
            print(f"Inpainting region {len(silence_regions) - i}/{len(silence_regions)}: "
                  f"{start/sr:.2f}s - {end/sr:.2f}s ({(end-start)/sr:.2f}s)")
            result = self.inpaint_region(result, sr, start, end)
        
        return result
    
    def _detect_silences(self, audio: np.ndarray, sr: int,
                         threshold_db: float, min_duration: float,
                         max_duration: float) -> List[Tuple[int, int]]:
        """Detect silence regions in audio."""
        # Convert to dB
        rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        rms_db = librosa.amplitude_to_db(rms + 1e-10)
        
        # Find frames below threshold
        silence_frames = rms_db < threshold_db
        
        # Convert to sample indices
        hop_length = 512
        silence_regions = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_silence:
                silence_start = i * hop_length
                in_silence = True
            elif not is_silent and in_silence:
                silence_end = i * hop_length
                duration = (silence_end - silence_start) / sr
                if min_duration <= duration <= max_duration:
                    silence_regions.append((silence_start, silence_end))
                in_silence = False
        
        # Handle silence at the end
        if in_silence:
            silence_end = len(audio)
            duration = (silence_end - silence_start) / sr
            if min_duration <= duration <= max_duration:
                silence_regions.append((silence_start, silence_end))
        
        return silence_regions
    
    def improve_transitions(self, audio: np.ndarray, sr: int,
                            transition_points: List[float],
                            transition_duration: float = 1.0) -> np.ndarray:
        """
        Improve transitions at specified points using generative smoothing.
        
        Args:
            audio: Input audio
            sr: Sample rate
            transition_points: List of transition times in seconds
            transition_duration: Duration around each point to process
            
        Returns:
            Audio with improved transitions
        """
        self.load_model()
        
        result = audio.copy()
        half_dur = transition_duration / 2
        
        for point in sorted(transition_points, reverse=True):
            start = max(0, int((point - half_dur) * sr))
            end = min(len(audio), int((point + half_dur) * sr))
            
            if end - start < sr * 0.1:  # Skip if too short
                continue
            
            print(f"Smoothing transition at {point:.2f}s")
            
            # Get context analysis
            analysis = self.analyze_context(audio, sr, start, end)
            
            # Generate transition-appropriate audio
            prompt = f"smooth transition, {analysis['prompt']}"
            
            # Generate and blend
            generated = self._generate_audio(prompt, duration_seconds=transition_duration)
            
            # Resample if needed
            if self.sample_rate != sr:
                generated = librosa.resample(generated, orig_sr=self.sample_rate, target_sr=sr)
            
            # Trim to length
            target_length = end - start
            if len(generated) > target_length:
                generated = generated[:target_length]
            elif len(generated) < target_length:
                generated = np.pad(generated, (0, target_length - len(generated)))
            
            # Blend with original (50/50 mix for smooth transition)
            original_region = result[start:end]
            blend_factor = 0.3  # 30% generated, 70% original
            result[start:end] = original_region * (1 - blend_factor) + generated * blend_factor
        
        return result
    
    def enhance_mashup(self, mashup_audio: np.ndarray, sr: int,
                       vocal_audio: np.ndarray = None,
                       segment_boundaries: List[float] = None,
                       inpaint_silences: bool = True,
                       smooth_transitions: bool = True) -> np.ndarray:
        """
        Full enhancement pipeline for mashup audio.
        
        Args:
            mashup_audio: The mashup to enhance
            sr: Sample rate
            vocal_audio: Optional separate vocal track for reference
            segment_boundaries: Optional list of segment boundary times
            inpaint_silences: Whether to inpaint detected silences
            smooth_transitions: Whether to smooth segment transitions
            
        Returns:
            Enhanced mashup audio
        """
        print("=" * 60)
        print("STABLE AUDIO MASHUP ENHANCEMENT")
        print("=" * 60)
        
        result = mashup_audio.copy()
        
        # Step 1: Inpaint silences
        if inpaint_silences:
            print("\nStep 1: Inpainting silences...")
            result = self.inpaint_silences(result, sr)
        
        # Step 2: Smooth transitions
        if smooth_transitions and segment_boundaries:
            print("\nStep 2: Smoothing transitions...")
            result = self.improve_transitions(result, sr, segment_boundaries)
        
        # Step 3: Final normalization
        print("\nStep 3: Final normalization...")
        max_val = np.max(np.abs(result))
        if max_val > 0.99:
            result = result * (0.99 / max_val)
        
        print("\nEnhancement complete!")
        return result


def create_inpainter(use_gpu: bool = True) -> StableAudioInpainter:
    """Factory function to create a Stable Audio inpainter."""
    return StableAudioInpainter(use_gpu=use_gpu)

