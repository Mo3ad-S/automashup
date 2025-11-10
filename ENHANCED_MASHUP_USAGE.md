# Enhanced Mashup Pipeline Usage

## Overview

The enhanced mashup pipeline addresses three main issues:
1. **Silences** - Detected and filled using audio inpainting
2. **Quality deterioration** - Enhanced using denoising, de-artifacting, and harmonic restoration
3. **Tempo/structure problems** - Fixed using beat-synchronous alignment and advanced tempo matching

## Quick Start

### Basic Usage

```python
import warnings
import soundfile as sf
from automashup.src.core.track import Track
from automashup.src.core.mashup_engine import MashupEngine
from automashup.src.enhanced_utils.config_loader import load_config

# Suppress warnings
warnings.filterwarnings("ignore")

# Load configuration
config = load_config()

# Initialize mashup engine
engine = MashupEngine(config=config)

# Load tracks (after preprocessing with allin1)
stored_data_path = "./automashup/data"
song_name_1 = "M2_lyrics"  # Song with vocals
song_name_2 = "M2_instru"  # Song with instrumental

# Create tracks
track_1 = Track.track_from_song(song_name_1, type='vocals', stored_data_path=stored_data_path)
track_2 = Track.track_from_song(song_name_2, type='bass', stored_data_path=stored_data_path)
track_3 = Track.track_from_song(song_name_2, type='drums', stored_data_path=stored_data_path)
track_4 = Track.track_from_song(song_name_2, type='other', stored_data_path=stored_data_path)

# Create mashup
tracks = [track_1, track_2, track_3, track_4]
mashup_result = engine.create_mashup(tracks)

# Save result
sf.write("enhanced_mashup.mp3", mashup_result.audio, mashup_result.sr)

print(f"Quality score: {mashup_result.quality_metrics.get('quality_score', 0.0):.2f}")
```

## Configuration

Edit `config.yaml` to customize the pipeline:

- `use_gpu`: Use GPU acceleration when available
- `use_phase_vocoder`: Use phase vocoder for better quality tempo/pitch changes
- `preserve_formants`: Preserve formants for vocal processing
- `inpainting_enabled`: Enable audio inpainting for silences
- `enhancement_enabled`: Enable quality enhancement
- `mixing_enabled`: Enable advanced mixing
- `mastering_enabled`: Enable final mastering
- `quality_threshold`: Quality threshold for reprocessing (0-1)

## Pipeline Steps

1. **Alignment & Synchronization** - Beat-synchronous alignment
2. **Tempo Matching** - Multi-stage tempo matching using phase vocoder
3. **Pitch Correction** - Formant-preserving pitch correction
4. **Silence Detection & Inpainting** - Detect and fill silences
5. **Quality Enhancement** - Denoising, de-artifacting, harmonic restoration
6. **Mixing** - Automatic EQ, compression, spatial processing
7. **Transition Smoothing** - Crossfade optimization
8. **Mastering** - Final EQ, limiting, loudness normalization
9. **Quality Check** - Artifact detection and quality metrics

## Backward Compatibility

The old `mashup_technic` functions are still available for backward compatibility. The new pipeline can be used alongside the old methods.

## Architecture

```
automashup/src/
├── core/                    # Core refactored components
│   ├── track.py             # Enhanced Track class
│   ├── segment.py            # Enhanced Segment class
│   └── mashup_engine.py      # Main mashup orchestrator
├── preprocessing/            # Enhanced preprocessing
│   ├── source_separator.py  # Multi-model separation
│   ├── quality_assessment.py
│   └── silence_detector.py
├── alignment/                # Enhanced alignment
│   ├── beat_sync.py
│   ├── tempo_matcher.py
│   └── pitch_corrector.py
├── inpainting/              # Audio inpainting
│   ├── silence_handler.py
│   └── audio_inpainter.py
├── enhancement/             # Quality enhancement
│   ├── enhancer.py
│   ├── vocal_enhancer.py
│   └── instrumental_enhancer.py
├── mixing/                  # Advanced mixing
│   ├── auto_mixer.py
│   ├── stem_mixer.py
│   └── transition_smoother.py
└── postprocessing/          # Final processing
    ├── mastering.py
    └── quality_checker.py
```

