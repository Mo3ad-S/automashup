from automashup.src.utils import increase_array_size
import librosa
import numpy as np
import pyrubberband as pyrb
import pyloudnorm as pyln
import librosa
try:
    from scipy.signal import butter, lfilter
except ImportError:  # Fallback if scipy is unavailable
    butter = lfilter = None

# Mashup Technics
# In this file, you may add mashup technics
# the input of such a method is a list of up to 4 objects of type Track. 
# You can modify them without making any copy, it's already done before.
# You may find useful methods in the track.py file
# Be sure to return a Track object

def mashup_technic(tracks, phase_fit=False, target_loudness=-14.0):
    # Mashup technic with first beat alignment and bpm sync
    sr = tracks[0].sr # The first track is used to determine the target bpm
    tempo = tracks[0].bpm
    main_track_length = len(tracks[0].audio)
    beginning = 0
    mashup = np.zeros(0)
    mashup_name = ""

    def _estimate_start_seconds(track):
        """Use downbeats/beats; otherwise start at 0 to avoid misalignment."""
        if hasattr(track, "downbeats") and getattr(track, "downbeats", []):
            return max(0.0, track.downbeats[0])
        if hasattr(track, "beats") and getattr(track, "beats", []):
            return max(0.0, track.beats[0])
        return 0.0

    def _butter_highpass(audio, cutoff_hz, sr, order=2):
        if butter is None or lfilter is None or len(audio) == 0:
            return audio
        nyq = 0.5 * sr
        normal_cutoff = cutoff_hz / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return lfilter(b, a, audio)

    def _normalize_peak(audio, target=0.9):
        if len(audio) == 0:
            return audio
        peak = np.max(np.abs(audio))
        if peak == 0:
            return audio
        if peak > target:
            audio = audio * (target / peak)
        return audio

    def _prep_stem(audio, sr, is_vocal=False):
        cutoff = 90.0 if is_vocal else 35.0
        audio = _butter_highpass(audio, cutoff, sr)
        audio = _normalize_peak(audio, target=0.9)
        return audio

    starts = [_estimate_start_seconds(t) for t in tracks]
    global_start = min(starts) if starts else 0.0

    # we add each track to the mashup
    for track in tracks:
        mashup_name += track.name + " " # name
        track_tempo = track.bpm
        track_beginning_temporal = _estimate_start_seconds(track)
        track_sr = track.sr
        track_beginning = track_beginning_temporal * track_sr
        track_audio = _prep_stem(track.audio, track_sr, is_vocal=(track == tracks[0]))

        # reset first beat position (and pad so all tracks line up at the earliest start)
        offset_sec = max(0.0, track_beginning_temporal - global_start)
        offset_samples = round(offset_sec * track_sr)
        track_audio_no_offset = np.concatenate([np.zeros(offset_samples), np.array(track_audio)[round(track_beginning):]])

        # Change the bpm if there is no phase fit
        if not phase_fit:
            track_audio_accelerated = pyrb.time_stretch(track_audio_no_offset, track_sr, rate = tempo / track_tempo)
        else:
            #bpm is handled in segment.py
            track_audio_accelerated = track_audio_no_offset

        # add the right number of zeros to align with the main track
        final_track_audio = np.concatenate((np.zeros(round(beginning)), track_audio_accelerated)) 

        size = max(len(mashup), len(final_track_audio))
        mashup = np.array(mashup)
        mashup = (increase_array_size(final_track_audio, size) + increase_array_size(mashup, size))

    # Adjust mashup length to be the same as the main track's audio length
    if len(mashup) > main_track_length:
        mashup = mashup[:main_track_length]
    else:
        mashup = increase_array_size(mashup, main_track_length)

    # Simple, stable ducking of instrument bus against vocals
    vocal = _prep_stem(tracks[0].audio, sr, is_vocal=True)
    vocal = increase_array_size(vocal, len(mashup))
    inst_bus = mashup - increase_array_size(tracks[0].audio, len(mashup))
    frame = 1024
    hop = 512
    vocal_env = librosa.feature.rms(y=vocal, frame_length=frame, hop_length=hop)[0]
    inst_env = librosa.feature.rms(y=inst_bus, frame_length=frame, hop_length=hop)[0]
    # Smooth envelope (attack/release)
    def _smooth(env, attack_ms=50, release_ms=180):
        sm = np.zeros_like(env)
        alpha_a = np.exp(-hop / (sr * attack_ms / 1000.0))
        alpha_r = np.exp(-hop / (sr * release_ms / 1000.0))
        prev = 0.0
        for i, e in enumerate(env):
            if e > prev:
                prev = alpha_a * prev + (1 - alpha_a) * e
            else:
                prev = alpha_r * prev + (1 - alpha_r) * e
            sm[i] = prev
        return sm
    vocal_env = _smooth(vocal_env)
    inst_env = _smooth(inst_env)
    ratio = np.where(inst_env > 1e-6, vocal_env / inst_env, 1.0)
    gain = np.clip(ratio, 0.5, 1.0)  # up to ~6 dB duck
    gain_samples = np.repeat(gain, hop)
    gain_samples = gain_samples[:len(inst_bus)]
    inst_bus = inst_bus * gain_samples
    mashup = vocal + inst_bus

    # Protect peak
    peak = np.max(np.abs(mashup)) if len(mashup) else 0
    if peak > 0.99:
        mashup = mashup * (0.99 / peak)

    # Apply LUFS normalization to the mashup
    meter = pyln.Meter(sr)  # Create a BS.1770 loudness meter
    mashup_loudness = meter.integrated_loudness(mashup)  # Measure the loudness
    print(f"Mashup loudness (before normalization): {mashup_loudness} LUFS")

    # Normalize the mashup to the target loudness (-14 LUFS by deafult)
    mashup_normalized = pyln.normalize.loudness(mashup, mashup_loudness, target_loudness)
    print(f"Mashup normalized to {target_loudness} LUFS")

    # we return a modified version of the first track
    # doing so, we keep its metadata
    tracks[0].audio = mashup

    return tracks[0]


def mashup_technic_repitch(tracks):
    # Mashup technique to change the key by repitch
    key = tracks[0].get_key() # target key
    for i in range(len(tracks)-1):
        tracks[i+1].pitch_track(key) # repitch

    return mashup_technic(tracks)


def mashup_technic_fit_phase(tracks):
    # Mashup technique with phase alignment (i.e., chorus with chorus, verse with verse...)
    # Each track's structure is aligned with the first one
    for i in range(len(tracks) - 1):
        tracks[i + 1].fit_phase(tracks[0])

    # Standard mashup method
    return mashup_technic(tracks, phase_fit=True)

def mashup_technic_fit_phase_repitch(tracks):
    # Mashup technique with phase alignment and repitch
    # Repitch has to be the last method for it to be effective
    key = tracks[0].get_key() # target key
    for i in range(len(tracks)-1):
        tracks[i + 1].fit_phase(tracks[0]) # phase fit
        tracks[i+1].pitch_track(key) # repitch
    # Phase fit mashup
    return mashup_technic(tracks, phase_fit=True)