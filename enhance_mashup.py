"""
Command-line script to enhance mashups using Stable Audio inpainting.

This script uses Stable Audio Open 1.0 to:
1. Detect and fill silence gaps
2. Smooth transitions between segments
3. Improve overall mashup quality

Usage:
    python enhance_mashup.py --input mashup.mp3 --output enhanced_mashup.mp3
    python enhance_mashup.py --input mashup.mp3 --inpaint-silences
    python enhance_mashup.py --input mashup.mp3 --smooth-transitions --boundaries 30,60,90,120
"""

import os
import sys
import argparse
import warnings
import numpy as np
import librosa
import soundfile as sf

# Suppress warnings
warnings.filterwarnings("ignore")

# Add project path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from automashup.src.inpainting.stable_audio_inpainter import StableAudioInpainter


def analyze_audio(audio: np.ndarray, sr: int):
    """Analyze audio and print statistics."""
    duration = len(audio) / sr
    rms = librosa.feature.rms(y=audio)[0]
    rms_db = librosa.amplitude_to_db(rms + 1e-10)
    
    # Detect silences
    silence_threshold = -40
    silence_frames = rms_db < silence_threshold
    
    # Count silence regions
    silence_count = 0
    total_silence_duration = 0
    in_silence = False
    silence_start = 0
    hop_length = 512
    
    for i, is_silent in enumerate(silence_frames):
        if is_silent and not in_silence:
            silence_start = i
            in_silence = True
        elif not is_silent and in_silence:
            duration_s = (i - silence_start) * hop_length / sr
            if duration_s > 0.3:  # Only count silences > 0.3s
                silence_count += 1
                total_silence_duration += duration_s
            in_silence = False
    
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Max amplitude: {np.max(np.abs(audio)):.4f}")
    print(f"  Mean RMS: {np.mean(rms):.4f}")
    print(f"  Significant silences (>0.3s): {silence_count}")
    print(f"  Total silence duration: {total_silence_duration:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description='Enhance mashups using Stable Audio inpainting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic enhancement (inpaint silences)
  python enhance_mashup.py --input mashup.mp3 --output enhanced.mp3
  
  # Only inpaint silences
  python enhance_mashup.py --input mashup.mp3 --inpaint-silences
  
  # Smooth transitions at specific times
  python enhance_mashup.py --input mashup.mp3 --smooth-transitions --boundaries 30,60,90,120
  
  # Full enhancement with custom thresholds
  python enhance_mashup.py --input mashup.mp3 --silence-threshold -35 --min-silence 0.5
  
  # Inpaint specific region
  python enhance_mashup.py --input mashup.mp3 --region 10.0,15.0 --prompt "120 BPM electronic drums"
        """
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path to input mashup file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Path to output file (default: input_enhanced.mp3)')
    
    # Enhancement options
    parser.add_argument('--inpaint-silences', action='store_true', default=True,
                        help='Detect and inpaint silence regions (default: True)')
    parser.add_argument('--no-inpaint-silences', action='store_false', dest='inpaint_silences',
                        help='Disable silence inpainting')
    parser.add_argument('--smooth-transitions', action='store_true', default=False,
                        help='Smooth transitions at segment boundaries')
    parser.add_argument('--boundaries', type=str, default=None,
                        help='Comma-separated transition times in seconds (e.g., "30,60,90")')
    
    # Silence detection parameters
    parser.add_argument('--silence-threshold', type=float, default=-40.0,
                        help='Silence detection threshold in dB (default: -40)')
    parser.add_argument('--min-silence', type=float, default=0.3,
                        help='Minimum silence duration to inpaint in seconds (default: 0.3)')
    parser.add_argument('--max-silence', type=float, default=5.0,
                        help='Maximum silence duration to inpaint in seconds (default: 5.0)')
    
    # Manual region inpainting
    parser.add_argument('--region', type=str, default=None,
                        help='Specific region to inpaint as "start,end" in seconds')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom prompt for region inpainting')
    
    # Processing options
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze audio without enhancement')
    
    args = parser.parse_args()
    
    # Set output path
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_enhanced{ext}"
    
    # Load audio
    print("=" * 60)
    print("STABLE AUDIO MASHUP ENHANCER")
    print("=" * 60)
    
    print(f"\nLoading: {args.input}")
    audio, sr = librosa.load(args.input, sr=None)
    
    print("\nInput audio analysis:")
    analyze_audio(audio, sr)
    
    if args.analyze_only:
        print("\nAnalysis complete (no enhancement performed)")
        return
    
    # Initialize inpainter
    print("\nInitializing Stable Audio inpainter...")
    inpainter = StableAudioInpainter(use_gpu=not args.no_gpu)
    
    # Process based on options
    result = audio.copy()
    
    # Manual region inpainting
    if args.region:
        start, end = map(float, args.region.split(','))
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        
        print(f"\nInpainting region: {start:.2f}s - {end:.2f}s")
        if args.prompt:
            print(f"Using prompt: {args.prompt}")
        
        result = inpainter.inpaint_region(
            result, sr, start_sample, end_sample,
            prompt=args.prompt,
            crossfade_duration=0.15
        )
    
    # Automatic silence inpainting
    elif args.inpaint_silences:
        print(f"\nInpainting silences (threshold: {args.silence_threshold} dB, "
              f"min: {args.min_silence}s, max: {args.max_silence}s)...")
        
        result = inpainter.inpaint_silences(
            result, sr,
            silence_threshold_db=args.silence_threshold,
            min_silence_duration=args.min_silence,
            max_silence_duration=args.max_silence
        )
    
    # Transition smoothing
    if args.smooth_transitions and args.boundaries:
        boundaries = [float(t) for t in args.boundaries.split(',')]
        print(f"\nSmoothing transitions at: {boundaries}")
        
        result = inpainter.improve_transitions(
            result, sr, boundaries,
            transition_duration=1.0
        )
    
    # Final normalization
    max_val = np.max(np.abs(result))
    if max_val > 0.99:
        result = result * (0.99 / max_val)
    
    # Save result
    print(f"\nSaving enhanced audio to: {args.output}")
    sf.write(args.output, result, sr)
    
    print("\nOutput audio analysis:")
    analyze_audio(result, sr)
    
    print("\n" + "=" * 60)
    print("ENHANCEMENT COMPLETE!")
    print("=" * 60)
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()

