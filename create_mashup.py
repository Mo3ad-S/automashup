"""
Command-line script to create mashups using the enhanced mashup pipeline.
Equivalent to the Tutorial notebook - make a mashup.ipynb

Usage:
    python create_mashup.py --vocals songs/M2_lyrics.mp3 --instrumental songs/M2_instru.mp3
    python create_mashup.py --vocals songs/M2_lyrics.mp3 --instrumental songs/M2_instru.mp3 --output mashup.mp3
    python create_mashup.py --vocals songs/M2_lyrics.mp3 --instrumental songs/M2_instru.mp3 --use-old-pipeline
"""

import os
import sys
import argparse
import warnings
import logging
import soundfile as sf
import numpy as np
import allin1
import automashup.src.utils as utils

# Import both old and new approaches
from automashup.src.track import Track  # Old approach
from automashup.src.core.track import Track as EnhancedTrack  # New enhanced approach
from automashup.src.core.mashup_engine import MashupEngine  # New enhanced pipeline
from automashup.src.enhanced_utils.config_loader import load_config
import automashup.src.mashup as mashupper  # Old approach

# Suppress natten deprecation chatter and other noisy warnings
logging.getLogger("natten").setLevel(logging.ERROR)
logging.getLogger("natten.functional").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="natten")
warnings.filterwarnings("ignore")


def preprocess_song(song_path, stored_data_path="./automashup/data", overwrite=False):
    """
    Preprocess a song using allin1.
    
    Args:
        song_path: Path to the song file
        stored_data_path: Path to store preprocessed data
        overwrite: Whether to overwrite existing analysis
    
    Returns:
        Song name (filename without extension)
    """
    print(f"Preprocessing: {song_path}")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        allin1.analyze(
            song_path,
            out_dir=f'{stored_data_path}/struct',
            demix_dir=f'{stored_data_path}/separated',
            keep_byproducts=True,
            overwrite=overwrite
        )
    
    # Find key
    utils.key_finder(song_path, stored_data_path=stored_data_path)
    
    # Extract song name
    song_name = os.path.splitext(os.path.basename(song_path))[0]
    print(f"  Preprocessing complete: {song_name}")
    
    return song_name


def create_mashup_old(tracks, output_path="mashup_old.mp3"):
    """
    Create mashup using the old pipeline.
    
    Args:
        tracks: List of Track objects
        output_path: Output file path
    
    Returns:
        Result track
    """
    print("\n" + "=" * 60)
    print("Using OLD mashup pipeline (mashup_technic_fit_phase_repitch)")
    print("=" * 60)
    
    mashup_result = mashupper.mashup_technic_fit_phase_repitch(tracks)
    
    # Save result
    sf.write(output_path, mashup_result.audio, mashup_result.sr)
    print(f"\nOLD mashup saved: {output_path}")
    print(f"  Audio length: {len(mashup_result.audio) / mashup_result.sr:.2f} seconds")
    print(f"  Max amplitude: {np.max(np.abs(mashup_result.audio)):.4f}")
    
    return mashup_result


def create_mashup_enhanced(tracks, config=None, output_path="mashup_enhanced.mp3"):
    """
    Create mashup using the enhanced pipeline.
    
    Args:
        tracks: List of Track objects
        config: Configuration dictionary (optional)
        output_path: Output file path
    
    Returns:
        Result track
    """
    print("\n" + "=" * 60)
    print("Using ENHANCED mashup pipeline")
    print("=" * 60)
    
    # Load configuration
    if config is None:
        config = load_config()
    
    # Initialize enhanced mashup engine
    engine = MashupEngine(config=config)
    
    # Convert tracks to enhanced Track objects
    enhanced_tracks = []
    for track in tracks:
        enhanced_track = EnhancedTrack(
            track_name=track.name,
            audio=track.audio.copy(),
            metadata={
                'bpm': getattr(track, 'bpm', 120),
                'beats': getattr(track, 'beats', []),
                'downbeats': getattr(track, 'downbeats', []),
                'segments': [{'start': s.start, 'end': s.end, 'label': s.label} for s in track.segments],
                'key': getattr(track, 'key', {}),
                'path': getattr(track, 'path', '')
            },
            sr=track.sr
        )
        enhanced_tracks.append(enhanced_track)
    
    # Create mashup
    mashup_result = engine.create_mashup(enhanced_tracks)
    
    # Save result
    sf.write(output_path, mashup_result.audio, mashup_result.sr)
    print(f"\nENHANCED mashup saved: {output_path}")
    print(f"  Audio length: {len(mashup_result.audio) / mashup_result.sr:.2f} seconds")
    print(f"  Max amplitude: {np.max(np.abs(mashup_result.audio)):.4f}")
    
    # Print quality metrics
    if hasattr(mashup_result, 'quality_metrics'):
        quality = mashup_result.quality_metrics
        print(f"  Quality score: {quality.get('quality_score', 0.0):.2f}")
        print(f"  SNR: {quality.get('snr', 0.0):.2f} dB")
        print(f"  RMS: {quality.get('rms', 0.0):.4f}")
        print(f"  Artifacts detected: {any(quality.get('artifacts_detected', {}).values())}")
        print(f"  Silences found: {len(quality.get('silences', []))}")
    
    return mashup_result


def compare_results(result_old, result_enhanced):
    """
    Compare old and enhanced mashup results.
    
    Args:
        result_old: Old pipeline result
        result_enhanced: Enhanced pipeline result
    """
    print("\n" + "=" * 60)
    print("QUALITY COMPARISON")
    print("=" * 60)
    
    print("\nOLD Pipeline:")
    print(f"  - Audio length: {len(result_old.audio) / result_old.sr:.2f} seconds")
    print(f"  - Max amplitude: {np.max(np.abs(result_old.audio)):.4f}")
    
    print("\nENHANCED Pipeline:")
    print(f"  - Audio length: {len(result_enhanced.audio) / result_enhanced.sr:.2f} seconds")
    print(f"  - Max amplitude: {np.max(np.abs(result_enhanced.audio)):.4f}")
    
    if hasattr(result_enhanced, 'quality_metrics'):
        quality = result_enhanced.quality_metrics
        print(f"  - Quality score: {quality.get('quality_score', 0.0):.2f}")
        print(f"  - SNR: {quality.get('snr', 0.0):.2f} dB")
        print(f"  - RMS: {quality.get('rms', 0.0):.4f}")
        print(f"  - Artifacts detected: {any(quality.get('artifacts_detected', {}).values())}")
        print(f"  - Silences found: {len(quality.get('silences', []))}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Create mashups using the enhanced mashup pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with preprocessing
  python create_mashup.py --vocals songs/M2_lyrics.mp3 --instrumental songs/M2_instru.mp3
  
  # Use preprocessed songs (skip preprocessing)
  python create_mashup.py --vocals-name M2_lyrics --instrumental-name M2_instru --skip-preprocess
  
  # Use only old pipeline
  python create_mashup.py --vocals songs/M2_lyrics.mp3 --instrumental songs/M2_instru.mp3 --use-old-pipeline
  
  # Use only enhanced pipeline
  python create_mashup.py --vocals songs/M2_lyrics.mp3 --instrumental songs/M2_instru.mp3 --use-enhanced-pipeline
  
  # Custom output path
  python create_mashup.py --vocals songs/M2_lyrics.mp3 --instrumental songs/M2_instru.mp3 --output my_mashup.mp3
        """
    )
    
    # Input options
    parser.add_argument('--vocals', type=str, help='Path to vocals song file')
    parser.add_argument('--instrumental', type=str, help='Path to instrumental song file')
    parser.add_argument('--vocals-name', type=str, help='Name of preprocessed vocals song (skip preprocessing)')
    parser.add_argument('--instrumental-name', type=str, help='Name of preprocessed instrumental song (skip preprocessing)')
    
    # Processing options
    parser.add_argument('--skip-preprocess', action='store_true', help='Skip preprocessing (use existing preprocessed data)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing preprocessed data')
    parser.add_argument('--use-old-pipeline', action='store_true', help='Use only old pipeline')
    parser.add_argument('--use-enhanced-pipeline', action='store_true', help='Use only enhanced pipeline')
    
    # Output options
    parser.add_argument('--output', type=str, default='mashup.mp3', help='Output file path (default: mashup.mp3)')
    parser.add_argument('--data-path', type=str, default='./automashup/data', help='Path to preprocessed data (default: ./automashup/data)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.skip_preprocess:
        if not args.vocals or not args.instrumental:
            parser.error("--vocals and --instrumental are required unless --skip-preprocess is used")
    else:
        if not args.vocals_name or not args.instrumental_name:
            parser.error("--vocals-name and --instrumental-name are required when using --skip-preprocess")
    
    stored_data_path = args.data_path
    
    # Preprocess songs if needed
    if not args.skip_preprocess:
        print("=" * 60)
        print("PREPROCESSING SONGS")
        print("=" * 60)
        
        song_name_1 = preprocess_song(args.vocals, stored_data_path, args.overwrite)
        song_name_2 = preprocess_song(args.instrumental, stored_data_path, args.overwrite)
    else:
        song_name_1 = args.vocals_name
        song_name_2 = args.instrumental_name
    
    # Load tracks
    print("\n" + "=" * 60)
    print("LOADING TRACKS")
    print("=" * 60)
    
    track_1 = Track.track_from_song(song_name_1, type='vocals', stored_data_path=stored_data_path)
    track_2 = Track.track_from_song(song_name_2, type='bass', stored_data_path=stored_data_path)
    track_3 = Track.track_from_song(song_name_2, type='drums', stored_data_path=stored_data_path)
    track_4 = Track.track_from_song(song_name_2, type='other', stored_data_path=stored_data_path)
    
    tracks = [track_1, track_2, track_3, track_4]
    
    print(f"\nTrack 1 (vocals): {track_1.name}")
    print(f"  BPM: {track_1.bpm}")
    print(f"  Key: {track_1.get_key()}")
    print(f"  Length: {len(track_1.audio) / track_1.sr:.2f} seconds")
    
    print(f"\nTrack 2 (bass): {track_2.name}")
    print(f"  BPM: {track_2.bpm}")
    print(f"  Length: {len(track_2.audio) / track_2.sr:.2f} seconds")
    
    print(f"\nTrack 3 (drums): {track_3.name}")
    print(f"  BPM: {track_3.bpm}")
    print(f"  Length: {len(track_3.audio) / track_3.sr:.2f} seconds")
    
    print(f"\nTrack 4 (other): {track_4.name}")
    print(f"  BPM: {track_4.bpm}")
    print(f"  Length: {len(track_4.audio) / track_4.sr:.2f} seconds")
    
    # Create mashups
    result_old = None
    result_enhanced = None
    
    if not args.use_enhanced_pipeline:
        # Create old mashup
        old_output = args.output.replace('.mp3', '_old.mp3') if not args.use_old_pipeline else args.output
        result_old = create_mashup_old(tracks, old_output)
    
    if not args.use_old_pipeline:
        # Create enhanced mashup
        enhanced_output = args.output.replace('.mp3', '_enhanced.mp3') if not args.use_enhanced_pipeline else args.output
        result_enhanced = create_mashup_enhanced(tracks, output_path=enhanced_output)
    
    # Compare results if both were created
    if result_old is not None and result_enhanced is not None:
        compare_results(result_old, result_enhanced)
    
    print("\n" + "=" * 60)
    print("MASHUP CREATION COMPLETE!")
    print("=" * 60)
    
    if result_old:
        print(f"OLD pipeline output: {old_output}")
    if result_enhanced:
        print(f"ENHANCED pipeline output: {enhanced_output}")


if __name__ == "__main__":
    main()

