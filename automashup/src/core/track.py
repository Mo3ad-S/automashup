import json
import librosa
import numpy as np
import copy
from typing import List, Dict, Optional, Any

try:
    from automashup.src.utils import note_to_frequency, calculate_pitch_shift, get_path, increase_array_size
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from automashup.src.utils import note_to_frequency, calculate_pitch_shift, get_path, increase_array_size

try:
    from automashup.src.core.segment import Segment
except ImportError:
    from automashup.src.segment import Segment


# Define a Track class to represent a musical track
# Enhanced with quality metrics, silence markers, and processing history
# This class uses objects of type Segment

class Track:
    transition_time = 1  # transition time in seconds

    # Standard constructor
    def __init__(self, track_name, audio, metadata, sr):
        # Initialize track properties
        self.name = track_name
        self.audio = audio
        self.sr = sr
        self.segments = []
        # Provide safe defaults so downstream code (segment linking, etc.) never
        # fails when optional metadata is missing.
        self.beats = list(metadata.get("beats", [])) if isinstance(metadata, dict) else []
        self.downbeats = list(metadata.get("downbeats", [])) if isinstance(metadata, dict) else []

        # Load track metadatas
        for key in metadata.keys():
            if key != "segments":
                setattr(self, key, metadata[key])
            else:
                for segment in metadata["segments"]:
                    # Create segment objects
                    if isinstance(segment, Segment):
                        segment_ = segment
                    else:
                        # we handle the segments as stored in the struct files
                        segment_ = Segment(segment)
                    segment_.link_track(self)
                    self.segments.append(segment_)
        
        # Enhanced features
        self.quality_metrics = {
            'snr': None,
            'rms': None,
            'zero_crossing_rate': None,
            'spectral_centroid': None,
            'artifacts_detected': False,
            'separation_confidence': None
        }
        
        self.silence_regions = []  # Global silence regions
        self.processing_history = []  # Track all processing operations
        
        # Track type (vocals, instrumental, etc.)
        self.track_type = self._infer_track_type(track_name)
        
        # Original audio backup
        self.original_audio = audio.copy() if audio is not None else None

    def _infer_track_type(self, track_name: str) -> str:
        """Infer track type from name."""
        name_lower = track_name.lower()
        if 'vocal' in name_lower:
            return 'vocals'
        elif 'instru' in name_lower or 'instrumental' in name_lower:
            return 'instrumental'
        elif 'bass' in name_lower:
            return 'bass'
        elif 'drum' in name_lower:
            return 'drums'
        elif 'other' in name_lower:
            return 'other'
        else:
            return 'mixed'

    @staticmethod
    def track_from_song(track_name, type, stored_data_path="."):
        # Function to create a track from a preprocessed song
        # type should be one of the following :
        # 'entire', 'bass', 'drums', 'vocals', 'other'
        name = track_name + ' - ' + type
        audio, sr = librosa.load(get_path(track_name, type, stored_data_path=stored_data_path), sr=None)
        struct_path = f"{stored_data_path}/struct/{track_name}.json"
        with open(struct_path, 'r') as file:
            metadata = json.load(file)
        track = Track(name, audio, metadata, sr)
        track._add_processing_step('track_from_song', {
            'track_name': track_name,
            'type': type,
            'stored_data_path': stored_data_path
        })
        return track

    def get_key(self):
        # Function to retrieve the key of a track
        # we use the "key" metadata which is a list of correlation
        # with each key. We look for the max correlation to get the
        # right key
        if not hasattr(self, 'key') or self.key is None:
            return None
        
        best_key, best_score = "", ""
        for key, score in self.key.items():
            if best_score == "" or best_score < score:
                best_key, best_score = key, score
        return best_key

    def __repitch(self, semitone_shift):
        # Function to repitch a track using a semitone shift
        # This will be replaced by pitch_corrector module
        # Keeping for backward compatibility
        import pyrubberband as pyrb
        shifted_audio = pyrb.pitch_shift(y=self.audio, sr=self.sr, n_steps=semitone_shift)
        self.audio = shifted_audio
        self._add_processing_step('repitch', {'semitone_shift': semitone_shift})

    def pitch_track(self, target_key):
        # Function to repitch a track to a target key
        # This will be enhanced by pitch_corrector module
        target_frequency = note_to_frequency(target_key)
        track_key = self.get_key()
        if track_key is None:
            return
        track_frequency = note_to_frequency(track_key)
        self.__repitch(calculate_pitch_shift(track_frequency, target_frequency))
        self._add_processing_step('pitch_track', {'target_key': target_key})

    def add_metronome(self, stored_data_path="."):
        # Function to add metronome sounds on the beats according
        # to the metadata of the track sounds
        downbeat_sound_audio, _ = librosa.load(f"{stored_data_path}/metronome-sounds/block.mp3")
        otherbeat_sound_audio, _ = librosa.load(f"{stored_data_path}/metronome-sounds/drumstick.mp3")

        # add sound for each beat
        for i, beat_frame in enumerate(self.beats):
            # if it's a downbeat, use the according sound
            clic_sound = downbeat_sound_audio if i % 4 == 0 else otherbeat_sound_audio
            clic = increase_array_size(clic_sound, len(self.audio[round(self.sr*beat_frame):]))
            # check that we do not get out of the track's bounds
            if len(self.audio[round(self.sr*beat_frame):]) >= len(clic):
                self.audio[round(self.sr*beat_frame):] += clic
        
        self._add_processing_step('add_metronome', {'stored_data_path': stored_data_path})

    def fit_phase(self, target_track):
        # Function to align track phases (verse, chorus, bridge, ...)
        # to a target track.
        # This will be enhanced by beat_sync module
        # Keeping original implementation for backward compatibility
        audio = np.array([])

        # lists of the return track beats and downbeats
        # we put 0 for convenience (see beats[-1] after)
        beats = [0]
        downbeats = [0]
        # keeps in memory the last segment to make it last longer in case we do not find a segment matching the next track segment  
        last_segment = self.segments[0] if len(self.segments) > 0 else None

        print(f" ********************** Adjusting the song {self.name}  **********************")

        # List of already found segments
        found_segments = {}

        # loop over each phase to reproduce
        for target_segment in target_track.segments:
            i = 0
            found_segment = False
            current_label = target_segment.label

            # Check if we have already found this label before
            if current_label in found_segments:
                start_index = found_segments[current_label] + 1
            else:
                start_index = 0

            i = start_index

            # loop over each segment to find occurrences
            while i < len(self.segments):
                segment = self.segments[i]
                if segment.label == current_label:
                    # If this is the first time we find the label or we have moved past the previous index
                    if not found_segment or i > found_segments[current_label]:
                        found_segment = True
                        tempo = round(len(segment.beats) / segment.duration) if segment.duration > 0 else 120
                        found_segments[current_label] = i
                        break
                i += 1

            # If no segment was found, reuse the last found segment
            if not found_segment and current_label in found_segments:
                last_found_index = found_segments[current_label]
                segment = self.segments[last_found_index]
                tempo = round(len(segment.beats) / segment.duration) if segment.duration > 0 else 120
                found_segment = True

            # if we do not find it, we add zeros with the right length
            if (not found_segment):
                tempo = round(len(target_segment.beats) / target_segment.duration) if target_segment.duration > 0 else 120
                try:
                    if len(target_segment.beats) > 0 and last_segment is not None:
                        segment = last_segment  
                        target_bpm = len(target_segment.beats) / target_segment.duration if target_segment.duration > 0 else 120

                        segment_fitted = segment.get_audio_beat_fitted(
                            len(target_segment.beats), 
                            target_bpm, 
                            len(target_segment.audio), 
                            self.sr
                        )
                        track_sr = target_track.sr
                        track_beginning_temporal = target_segment.beats[0] if len(target_segment.beats) > 0 else 0
                        track_beginning = track_beginning_temporal * track_sr
                        pad_len = max(0, round(track_beginning) - len(audio))
                        if pad_len > 0:
                            audio = np.concatenate([np.zeros(pad_len), audio])
                        audio = np.concatenate([audio, segment_fitted.audio])

                        # we add the new beats to be able to sync after
                        beats += [beats[-1] + phase_beat for phase_beat in segment_fitted.beats]
                        downbeats += [downbeats[-1] + phase_downbeat for phase_downbeat in segment_fitted.downbeats]
                except Exception as e:
                    print(f"Error fitting silence. Error: {e}")
            else:
                try:
                    # if we find it, we make it fit to the desired beat number
                    if len(target_segment.beats) > 0:
                        last_segment = segment
                        target_bpm = len(target_segment.beats) / target_segment.duration if target_segment.duration > 0 else 120

                        segment_fitted = segment.get_audio_beat_fitted(
                            len(target_segment.beats), 
                            target_bpm, 
                            len(target_segment.audio), 
                            self.sr
                        )
                        track_sr = target_track.sr
                        track_beginning_temporal = target_segment.beats[0] if len(target_segment.beats) > 0 else 0
                        track_beginning = track_beginning_temporal * track_sr
                        pad_len = max(0, round(track_beginning) - len(audio))
                        if pad_len > 0:
                            audio = np.concatenate([np.zeros(pad_len), audio])
                        audio = np.concatenate([audio, segment_fitted.audio])

                        # we add the new beats to be able to sync after
                        beats += [beats[-1] + phase_beat for phase_beat in segment_fitted.beats]
                        downbeats += [downbeats[-1] + phase_downbeat for phase_downbeat in segment_fitted.downbeats]
                except Exception as e:
                    print(f"Error fitting segment: {segment.label if segment else 'unknown'} at {segment.start if segment else 'unknown'}, Error: {e}")
                    continue

        # we get rid of the first beats added for convenience
        beats = beats[1:]
        downbeats = downbeats[1:]

        self.audio = audio
        self.beats = beats
        self.downbeats = downbeats
        
        self._add_processing_step('fit_phase', {'target_track': target_track.name})

    def _add_processing_step(self, operation: str, parameters: Dict[str, Any]):
        """Add a processing step to the history."""
        self.processing_history.append({
            'operation': operation,
            'parameters': parameters,
            'timestamp': None
        })

    def add_silence_region(self, start: float, end: float, silence_type: str = 'gap'):
        """Add a silence region marker."""
        self.silence_regions.append((start, end))

    def update_quality_metrics(self, metrics: Dict[str, Any]):
        """Update quality metrics for this track."""
        self.quality_metrics.update(metrics)

    def get_quality_score(self) -> float:
        """Calculate an overall quality score from metrics."""
        if self.quality_metrics['snr'] is None:
            return 0.0
        
        score = 0.0
        if self.quality_metrics['snr'] is not None:
            score += min(self.quality_metrics['snr'] / 30.0, 1.0) * 0.4
        if self.quality_metrics['rms'] is not None:
            score += min(self.quality_metrics['rms'] * 10, 1.0) * 0.3
        if not self.quality_metrics['artifacts_detected']:
            score += 0.3
        
        return min(score, 1.0)

    @staticmethod
    def get_segments(track_name, stored_data_path="."):
        # This method should return a list of segments for the given song
        track = Track.track_from_song(track_name, 'entire', stored_data_path=stored_data_path)
        return [segment.label for segment in track.segments]

    @staticmethod
    def get_segments_full(track_name, stored_data_path="."):
        # This method should return a list of the structure of segments for the given song
        track = Track.track_from_song(track_name, 'entire', stored_data_path=stored_data_path)
        return [{'start': segment.start, 'end': segment.end, 'label': segment.label} for segment in track.segments]
    
    def __deepcopy__(self, memo):
        """Custom deepcopy implementation for Track objects."""
        # Create a new Track with copied attributes
        new_track = Track.__new__(Track)
        memo[id(self)] = new_track
        
        # Copy all attributes
        for key, value in self.__dict__.items():
            if key == 'audio' or key == 'original_audio':
                # Copy numpy arrays
                setattr(new_track, key, value.copy() if value is not None else None)
            elif key == 'segments':
                # Deep copy segments
                new_track.segments = [copy.deepcopy(seg, memo) for seg in value]
            else:
                # Deep copy other attributes
                setattr(new_track, key, copy.deepcopy(value, memo))
        
        return new_track

