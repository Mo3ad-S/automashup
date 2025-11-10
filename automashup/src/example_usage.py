"""
Example usage of the enhanced mashup pipeline.
"""
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

# Example: Create mashup from two songs
# First song provides vocals, second provides instrumental

# Preprocess songs (using allin1 - this should be done separately)
# allin1.analyze("song1.mp3", out_dir="./data/struct", demix_dir="./data/separated")
# allin1.analyze("song2.mp3", out_dir="./data/struct", demix_dir="./data/separated")

# Load tracks
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

print(f"Mashup saved! Quality score: {mashup_result.quality_metrics.get('quality_score', 0.0):.2f}")

