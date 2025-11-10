import os
import warnings
import soundfile as sf

import allin1
from IPython.display import Audio

import automashup.src.utils as utils
from automashup.src.track import Track
import automashup.src.mashup as mashupper

warnings.filterwarnings("ignore", category=UserWarning, module="natten")

folder_path = "/home/mo3ad/am/songs"
stored_data_path = "/home/mo3ad/am/automashup/data"

# First song to analyze
song_path = f"{folder_path}/Bam Bam - Hi-Q.mp3" # Path to the song
with warnings.catch_warnings():
    warnings.simplefilter("ignore") # Actually doesn't work...
    allin1.analyze(song_path, out_dir=f'{stored_data_path}/struct', demix_dir=f'{stored_data_path}/separated', keep_byproducts=True, overwrite=False)
utils.key_finder(song_path, stored_data_path=stored_data_path)

# Second song to analyze
song_path = f"{folder_path}/The Oranges Band - Ride The Nuclear Wave.mp3" # Path to the song
with warnings.catch_warnings():
    warnings.simplefilter("ignore") # Actually doesn't work...
    allin1.analyze(song_path, out_dir=f'{stored_data_path}/struct', demix_dir=f'{stored_data_path}/separated', keep_byproducts=True, overwrite=False)
utils.key_finder(song_path, stored_data_path=stored_data_path)

song_name_1 = 'Bam Bam - Hi-Q'
song_name_2 = 'The Oranges Band - Ride The Nuclear Wave'

tracks =  [] # input of the mashup methods

# type attribute enables to choose a separated part of a song (from demucs source separation)
# it can be 'vocals', 'bass', 'drums' or 'other'
track_1 = Track.track_from_song(song_name_1, type='vocals', stored_data_path=stored_data_path)
track_2 = Track.track_from_song(song_name_2, type='bass', stored_data_path=stored_data_path)
track_3 = Track.track_from_song(song_name_2, type='drums', stored_data_path=stored_data_path)
track_4 = Track.track_from_song(song_name_2, type='other', stored_data_path=stored_data_path)

tracks = [track_1, track_2, track_3, track_4]


print(f"BPM : {track_1.bpm}")
print(f"Key correlation : {track_1.key}")
print(f"Beat frames : {track_1.beats}")
print(f"Track audio : {track_1.audio}")
print(f"Track Sampling Frequency {track_1.sr}")

mashup_result = mashupper.mashup_technic_fit_phase_repitch(tracks) # Apply the mashup_technic function to the 'tracks' list.


Audio(mashup_result.audio, rate = mashup_result.sr)


sf.write("mashup.mp3", mashup_result.audio, mashup_result.sr)