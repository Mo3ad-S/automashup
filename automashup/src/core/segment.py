import numpy as np
import copy
from typing import List, Dict, Optional, Any
try:
    from automashup.src.utils import closest_index
except ImportError:
    # Fallback for backward compatibility
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from automashup.src.utils import closest_index

# Define a Segment class to represent a part of a track
# Enhanced with quality metrics, silence markers, and processing history

class Segment:
    transition_time = 0.5  # transition time in seconds

    def __init__(self, segment_dict):
        # We create a segment from a dict coming from metadata.
        # They look like this : 
        # {
        #   "start": 0.4,
        #   "end": 22.82,
        #   "label": "verse"
        # }
        
        for key in segment_dict.keys():
            setattr(self, key, segment_dict[key])

        # We calculate the time in minutes for each segment
        self.duration = (self.end - self.start) / 60
        
        # Enhanced features
        self.quality_metrics = {
            'snr': None,  # Signal-to-noise ratio
            'rms': None,  # Root mean square energy
            'zero_crossing_rate': None,
            'spectral_centroid': None,
            'artifacts_detected': False
        }
        
        self.silence_regions = []  # List of (start, end) tuples in seconds relative to segment start
        self.silence_markers = []  # Detailed silence information
        
        self.processing_history = []  # Track all processing operations applied
        
        # Initialize empty arrays (will be populated in link_track)
        self.audio = np.array([])
        self.beats = []
        self.downbeats = []
        self.left_transition = np.array([])
        self.right_transition = np.array([])
        self.sr = None

    def link_track(self, track):
        # Function to link a segment to a track, must be set whenever
        # we create a segment
        # it loads all the pieces of information useful for a segment
        self.sr = track.sr
        beats = track.beats
        downbeats = track.downbeats
        
        start_beat = closest_index(self.start, beats)
        end_beat = closest_index(self.end, beats)

        self.beats = beats[start_beat:end_beat]

        if self.beats == []:
            self.downbeats = []
            self.audio = np.array([])
            self.left_transition = np.array([])
            self.right_transition = np.array([])
        else:
            # Make sure the first beat and downbeat starts on zero 
            self.beats = self.beats - np.repeat(self.beats[0], len(self.beats))

            self.downbeats = downbeats - np.repeat(downbeats[0], len(downbeats))
            self.downbeats = [downbeat for downbeat in self.downbeats if downbeat in self.beats]
            if self.downbeats != []:
                self.downbeats = self.downbeats - np.repeat(self.downbeats[0], len(self.downbeats))

            self.audio = track.audio[round(self.start*self.sr):round(self.end*self.sr)]

            if self.start - self.transition_time > 0:
                self.left_transition = track.audio[round((self.start-self.transition_time)*self.sr):round(self.start*self.sr)]
            else:
                self.left_transition = track.audio[:round(self.start*self.sr)]
            if self.end + self.transition_time < len(track.audio) / self.sr:
                self.right_transition = track.audio[round(self.end*self.sr):round((self.end+self.transition_time)*self.sr)]
            else: 
                self.right_transition = track.audio[round(self.end*self.sr):]
        
        # Record initial linking in processing history
        self._add_processing_step('link_track', {'track_name': track.name})

    def concatenate(self):
        # Calculate the seconds to shift the incoming beats to start where the current segment's audio ends.
        offset = len(self.audio) / self.sr

        # Concatenate the beats array of the current segment
        new_beats = self.beats + offset
        self.beats = np.concatenate([self.beats, new_beats])

        # Concatenate the downbeats array
        new_downbeats = self.downbeats + offset
        self.downbeats = np.concatenate([self.downbeats, new_downbeats])

        # Concatenate the audio data of the two segments
        self.audio = np.concatenate((self.audio, self.audio))
        
        # Record concatenation in processing history
        self._add_processing_step('concatenate', {})

    def get_audio_beat_fitted(self, beat_number, tempo, duration, sr):
        """
        Adjusts the audio segment to fit a specified number of beats at a given tempo.

        Parameters:
        - beat_number: The target number of beats for the segment.
        - tempo: The target tempo (BPM) for the segment.
        - duration: The target segment duration in samples

        Returns:
        - A new audio segment adjusted to the specified number of beats and tempo.
        """
        try:
            # Make a deep copy of the segment to avoid modifying the original
            result = copy.deepcopy(self)

            if beat_number == 0:
                # If beat_number is 0, return an empty segment
                result.audio = np.array([])
                result.beats = []
                result.downbeats = []
            else:
                # We compare the bpm of the target segment and the current segment. 
                # If the difference is too big, half or double it
                segment_bpm = (len(result.beats) / result.duration) if result.duration > 0 else tempo
                original_tempo = tempo
                
                if segment_bpm > 0 and tempo / segment_bpm > 1.5:
                    tempo /= 2
                    beat_number //= 2
                elif segment_bpm > 0 and tempo / segment_bpm < 0.75:
                    tempo *= 2
                    beat_number //= 2

                # We calculate the rate of stretch for the segment.
                stretch_rate = tempo / segment_bpm if segment_bpm > 0 else 1.0
                
                # Store original audio for quality comparison
                original_audio = result.audio.copy()
                
                # Time stretching will be done by external tempo_matcher module
                # For now, we just prepare the segment
                result.stretch_rate = stretch_rate
                result.target_tempo = tempo
                result.target_beats = beat_number
                result.target_duration = duration

                # We concatenate the segment to itself if it's shorter than the target
                while len(result.beats) < beat_number:
                    print(f"Concatenating for segment '{result.label}'")
                    result.concatenate()

                # Then we cut the amount of beats and downbeats so they end at the same time
                result.beats = result.beats[:beat_number]
                if len(result.beats) > 0:
                    result.downbeats = [downbeat for downbeat in result.downbeats if downbeat <= result.beats[-1]]
                else:
                    result.downbeats = []

                # Then we cut the audio for the same length as the objective
                if len(result.audio) > duration:
                    result.audio = result.audio[:duration]
                else:
                    # Pad if necessary
                    padding = duration - len(result.audio)
                    result.audio = np.pad(result.audio, (0, padding), mode='constant')
                
                # Record processing
                result._add_processing_step('get_audio_beat_fitted', {
                    'original_tempo': original_tempo,
                    'target_tempo': tempo,
                    'stretch_rate': stretch_rate,
                    'beat_number': beat_number
                })

            return result

        except Exception as e:
            # Handle any exceptions gracefully and print detailed error message
            print(f"Error fitting segment '{self.label}' to {beat_number} beats: {e}")
            raise e

    def _add_processing_step(self, operation: str, parameters: Dict[str, Any]):
        """Add a processing step to the history."""
        self.processing_history.append({
            'operation': operation,
            'parameters': parameters,
            'timestamp': None  # Could add timestamp if needed
        })

    def add_silence_region(self, start: float, end: float, silence_type: str = 'gap'):
        """Add a silence region marker."""
        self.silence_regions.append((start, end))
        self.silence_markers.append({
            'start': start,
            'end': end,
            'type': silence_type,  # 'gap', 'transition', 'missing_segment'
            'duration': end - start
        })

    def update_quality_metrics(self, metrics: Dict[str, Any]):
        """Update quality metrics for this segment."""
        self.quality_metrics.update(metrics)

    def get_quality_score(self) -> float:
        """Calculate an overall quality score from metrics."""
        # Simple weighted average (can be improved)
        if self.quality_metrics['snr'] is None:
            return 0.0
        
        score = 0.0
        if self.quality_metrics['snr'] is not None:
            score += min(self.quality_metrics['snr'] / 30.0, 1.0) * 0.4  # Normalize SNR
        if self.quality_metrics['rms'] is not None:
            score += min(self.quality_metrics['rms'] * 10, 1.0) * 0.3
        if not self.quality_metrics['artifacts_detected']:
            score += 0.3
        
        return min(score, 1.0)

