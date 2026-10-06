#!/usr/bin/env python3
"""Polyphonic, MIDI-driven Pratt filter/wavetable synthesizer.

Usage: python3 pratt_midi_synth.py input.mid --output result.mp3
No soundfont and no MIDI library are required. This is an offline renderer.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy import signal
from midi_file import read_midi, polyphony

FS = 44100
TABLE_SIZE = 4096
ROOM_TAIL = 3.2
FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def factors(n):
    result = {}
    p = 2
    while p * p <= n:
        while n % p == 0:
            result[p] = result.get(p, 0) + 1
            n //= p
        p += 1
    if n > 1:
        result[n] = result.get(n, 0) + 1
    return result


def multiply(a, b):
    c = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            c[i+j] += ai * bj
    return tuple(c)


@lru_cache(None)
def pratt(n):
    if n == 1:
        return (1,)
    if n == 2:
        return (0, 1)
    if factors(n) == {n: 1}:
        a = list(pratt(n-1))
        a[0] += 1
        return tuple(a)
    a = (1,)
    for p, k in factors(n).items():
        for _ in range(k):
            a = multiply(a, pratt(p))
    return a


def response(n, coordinate):
    """H_n at the dimensionless frequency coordinate s/omega0=i*coordinate."""
    return n / np.polynomial.polynomial.polyval(2 + 1j * coordinate, pratt(n))


PRESETS = {
    "piano": dict(base=5, roll=0.92, xi=0.15, attack=0.0035, release=0.46,
                  decay=1.35, bright_decay=0.38, level=0.78, vibrato=0.0),
    "electric_piano": dict(base=11, roll=0.90, xi=0.135, attack=0.0045, release=0.38,
                           decay=1.6, bright_decay=0.65, level=0.76, vibrato=0.0),
    "organ": dict(base=2, roll=1.05, xi=0.13, attack=0.009, release=0.12,
                  decay=0.0, bright_decay=0.0, level=0.66, vibrato=0.0007),
    "pluck": dict(base=17, roll=0.78, xi=0.12, attack=0.0025, release=0.23,
                  decay=0.75, bright_decay=0.25, level=0.78, vibrato=0.0),
    "bass": dict(base=3, roll=1.02, xi=0.16, attack=0.008, release=0.24,
                 decay=1.25, bright_decay=0.28, level=0.86, vibrato=0.0),
    "strings": dict(base=7, roll=0.72, xi=0.115, attack=0.018, release=0.16,
                    decay=0.0, bright_decay=0.0, level=0.65, vibrato=0.0014),
    "brass": dict(base=11, roll=0.66, xi=0.11, attack=0.017, release=0.14,
                  decay=0.0, bright_decay=0.0, level=0.68, vibrato=0.0009),
    "reed": dict(base=5, roll=0.80, xi=0.11, attack=0.012, release=0.11,
                 decay=0.0, bright_decay=0.0, level=0.67, vibrato=0.0008),
    "flute": dict(base=3, roll=1.75, xi=0.15, attack=0.018, release=0.13,
                  decay=0.0, bright_decay=0.0, level=0.75, vibrato=0.0010),
    "pad": dict(base=37, roll=0.82, xi=0.10, attack=0.034, release=0.42,
                decay=0.0, bright_decay=0.0, level=0.58, vibrato=0.0013),
    "timpani": dict(base=7, roll=1.35, xi=0.19, attack=0.0015, release=0.32,
                    decay=0.62, bright_decay=0.14, level=0.93, vibrato=0.0),
}


def preset_name(program):
    if program == 47:
        return "timpani"
    if program < 4:
        return "piano"
    if 4 <= program <= 5:
        return "electric_piano"
    if program < 16:
        return "pluck"
    if program < 24:
        return "organ"
    if program < 32:
        return "pluck"
    if program < 40:
        return "bass"
    if program in (45, 46):
        return "pluck"
    if program < 50:
        return "strings"
    if program < 56:
        return "pad"
    if program < 64:
        return "brass"
    if program < 72:
        return "reed"
    if program < 80:
        return "flute"
    return "pad"


@lru_cache(maxsize=4096)
def wavetable(pitch, name, velocity_bucket, dark=False):
    patch = PRESETS[name]
    f0 = 440.0 * 2 ** ((pitch - 69) / 12)
    harmonics = min(56, int((FS * 0.42) // f0))
    k = np.arange(1, harmonics + 1)
    n = (pitch + 1) * patch["base"]
    velocity = (velocity_bucket + 0.5) / 8
    xi = patch["xi"] * (1.20 - 0.30 * velocity)
    z = response(n, xi * k)
    amplitudes = k ** (-patch["roll"] - (0.75 if dark else 0))
    if name == "reed":
        amplitudes *= np.where(k % 2, 1.0, 0.32)
    if name == "flute":
        amplitudes *= np.where(k > 3, 0.35, 1.0)
    spectrum = np.zeros(TABLE_SIZE // 2 + 1, dtype=complex)
    # A sine-rich excitation with the exact complex Pratt filter response.
    spectrum[k] = -1j * amplitudes * z
    table = np.fft.irfft(spectrum, TABLE_SIZE).astype(np.float32)
    table /= max(float(np.max(abs(table))), 1e-20)
    return np.concatenate((table, table[:1])), n, harmonics


def lookup(table, phase):
    position = np.remainder(phase, 1.0) * TABLE_SIZE
    index = position.astype(np.int32)
    fraction = position-index
    return (table[index] + fraction * (table[index+1]-table[index])).astype(np.float32)


def automation(events, times, default):
    if not events:
        return np.full_like(times, default, dtype=np.float32)
    # Duplicate-time messages retain the final state, then a short smoothing
    # ramp suppresses controller clicks without rewriting the MIDI timeline.
    unique = {}
    for t, value in events:
        unique[t] = value
    ts = np.asarray(sorted(unique), dtype=float)
    vs = np.asarray([unique[t] for t in ts], dtype=float)
    if len(ts) == 1:
        return np.full_like(times, vs[0], dtype=np.float32)
    knots_t = []
    knots_v = []
    previous = vs[0]
    for i, (t, v) in enumerate(zip(ts, vs)):
        if i:
            knots_t.append(max(ts[i-1]+1e-8, t-0.006))
            knots_v.append(previous)
        knots_t.append(t)
        knots_v.append(v)
        previous = v
    return np.interp(times, knots_t, knots_v).astype(np.float32)


def melodic_voice(note, channel_controls, max_end):
    name = preset_name(note.program)
    patch = PRESETS[name]
    held = max(0.003, note.end - note.start)
    # Freely decaying piano/plucked modes fall below -80 dB naturally.
    active_duration = held + patch["release"] * 2.3
    if patch["decay"]:
        active_duration = min(active_duration, patch["decay"] * 9.6)
    active_duration = min(active_duration, max_end-note.start)
    count = max(16, int(math.ceil(active_duration * FS)))
    t = np.arange(count, dtype=np.float32) / FS
    f0 = 440.0 * 2 ** ((note.pitch - 69) / 12)
    bends = channel_controls["bend"]
    if len(bends) == 1 and abs(bends[0][1]) < 1e-12:
        phase = f0 * t
    else:
        bend = automation(bends, note.start+t, 0)
        phase = np.cumsum(f0 * 2 ** (bend/12) / FS, dtype=np.float64)
    phase += (note.channel * 0.071 + note.track * 0.011) % 1
    if patch["vibrato"]:
        rate = 5.1 + 0.05 * note.channel
        depth = patch["vibrato"]
        phase += f0 * depth / (2*np.pi*rate) * (1 - np.cos(2*np.pi*rate*t))
    bucket = min(7, note.velocity // 16)
    table, n, harmonics = wavetable(note.pitch, name, bucket, False)
    wave = lookup(table, phase)
    if patch["bright_decay"]:
        dark, _, _ = wavetable(note.pitch, name, bucket, True)
        bright = np.exp(-t/patch["bright_decay"])
        wave = wave * bright + lookup(dark, phase) * (1-bright)
    attack = min(patch["attack"], max(0.002, held*0.35))
    env = np.minimum(t / attack, 1.0)
    env = env * env * (3-2*env)
    if patch["decay"]:
        env *= np.exp(-t / patch["decay"])
    release_time = np.maximum(t-held, 0)
    env *= np.exp(-release_time / patch["release"])
    # The final 12 ms always approach zero, even when an audible tail is
    # deliberately bounded for an offline rendering.
    fade = min(int(0.012*FS), count//4)
    env[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32) ** 2
    wave *= env * (note.velocity/127) ** 1.35 * patch["level"]
    return wave, n, harmonics


def percussion_voice(note, max_end):
    duration = min(1.2 if note.pitch in (46, 49, 51, 57) else 0.45, max_end-note.start)
    count = max(16, int(duration*FS))
    t = np.arange(count)/FS
    rng = np.random.default_rng(note.pitch * 131 + note.velocity)
    noise = rng.standard_normal(count)
    if note.pitch in (35, 36):
        phase = 2*np.pi*(48*t+4*(1-np.exp(-20*t)))
        wave = np.sin(phase)*np.exp(-t/0.10) + 0.08*noise*np.exp(-t/0.008)
    else:
        frequency = np.fft.rfftfreq(count, 1/FS)
        shaped = np.fft.rfft(noise) * response(note.pitch+1, frequency/6000)
        shaped[frequency < (4500 if note.pitch in (42, 44, 46) else 250)] *= 0.08
        wave = np.fft.irfft(shaped, count)*np.exp(-t/(0.24 if note.pitch == 46 else 0.075))
    wave /= max(np.max(abs(wave)), 1e-10)
    wave *= 0.72*(note.velocity/127)**1.25
    wave[:80] *= np.linspace(0, 1, 80)**2
    wave[-160:] *= np.linspace(1, 0, 160)**2
    return wave.astype(np.float32), note.pitch+1, 0


def render_midi(midi_path, output_path, title=None, composer=None, max_seconds=None):
    if not FFMPEG or not FFPROBE:
        raise RuntimeError("Install FFmpeg and make ffmpeg and ffprobe available on PATH.")
    began = time.monotonic()
    midi = read_midi(midi_path)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = output_path.parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    duration = max(midi["duration"], max(n.end for n in midi["notes"])) + ROOM_TAIL
    if max_seconds:
        duration = min(duration, max_seconds)
    total = int(math.ceil(duration * FS))
    mix = np.zeros((total, 2), dtype=np.float32)
    groups = defaultdict(list)
    for note in midi["notes"]:
        if note.start < duration:
            groups[note.channel].append(note)
    used_models = {}
    rendered = 0
    for channel, notes in sorted(groups.items()):
        bus = np.zeros(total, dtype=np.float32)
        for note in notes:
            if channel == 9:
                wave, n, harmonic_count = percussion_voice(note, duration)
                family = "percussion"
            else:
                wave, n, harmonic_count = melodic_voice(note, midi["controls"][channel], duration)
                family = preset_name(note.program)
            start = int(round(note.start*FS))
            end = min(total, start+len(wave))
            bus[start:end] += wave[:end-start]
            rendered += 1
            used_models[(note.pitch, family)] = {"midi_pitch": note.pitch, "family": family,
                                                 "pratt_index": n, "degree": len(pratt(n))-1,
                                                 "harmonics": harmonic_count,
                                                 "f_ascending_coefficients": list(pratt(n))}
        cc = midi["controls"][channel]
        block = 131072
        for start in range(0, total, block):
            end = min(total, start+block)
            times = np.arange(start, end)/FS
            volume = automation(cc[7], times, 100)/127
            expression = automation(cc[11], times, 127)/127
            pan = automation(cc[10], times, 64)/127
            angle = pan * np.pi/2
            audio = bus[start:end] * volume**1.25 * expression**1.1
            mix[start:end, 0] += audio * np.cos(angle)
            mix[start:end, 1] += audio * np.sin(angle)
        del bus
        print(f"{output_path.stem}: channel {channel+1}, {len(notes)} notes", flush=True)
    assert rendered == sum(len(x) for x in groups.values())
    assert np.all(np.isfinite(mix))

    # Shared diffuse stereo room. Apply the room to the actual complete
    # performance; no pre-existing recording or additional music is mixed in.
    # A mono send is sufficient and keeps memory bounded for long MIDI files.
    room_send = 0.5 * (mix[:, 0]+mix[:, 1])
    low = signal.butter(1, 3800, fs=FS, output="sos")
    room_send = signal.sosfilt(low, room_send).astype(np.float32)
    taps = [(0.053, .115, 0), (0.071, .112, 1), (0.113, .080, 1),
            (0.149, .076, 0), (0.199, .052, 0), (0.251, .052, 1),
            (0.337, .037, 1), (0.421, .034, 0), (0.557, .024, 0),
            (0.683, .022, 1), (0.883, .016, 0), (1.091, .013, 1),
            (1.433, .008, 0), (1.701, .007, 1)]
    for seconds, level, side in taps:
        shift = int(seconds*FS)
        if shift < total:
            mix[shift:, side] += level * room_send[:-shift]
    del room_send
    highpass = signal.butter(1, 22, "highpass", fs=FS, output="sos")
    for side in range(2):
        mix[:, side] = signal.sosfilt(highpass, mix[:, side])
    fade = min(int(.42*FS), total//10)
    mix[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)[:, None]**2
    # Preserve dynamics with one linear master gain, leaving MP3 headroom.
    raw_peak = float(np.max(abs(mix)))
    assert raw_peak > 0.001
    master_gain = 0.78/raw_peak
    mix *= master_gain

    pending = output_path.with_name(output_path.stem + ".encoding.mp3")
    ffmpeg = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
              "-f", "f32le", "-ar", str(FS), "-ac", "2", "-i", "pipe:0",
              "-codec:a", "libmp3lame", "-b:a", "192k", "-id3v2_version", "3",
              "-metadata", f"title={title or output_path.stem}",
              "-metadata", "artist=Pratt-Synthesizer", "-metadata", "album=Pratt MIDI",
              "-metadata", f"composer={composer or ''}",
              "-metadata", f"comment=Source MIDI: {Path(midi_path).name}; Pratt harmonic filter synthesis.", str(pending)]
    process = subprocess.Popen(ffmpeg, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for start in range(0, total, 131072):
            process.stdin.write(mix[start:start+131072].astype("<f4", copy=False).tobytes())
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RuntimeError(error)
    except Exception:
        process.kill()
        raise
    with pending.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(pending, output_path)
    probe = json.loads(subprocess.run([FFPROBE, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(output_path)],
                                      check=True, capture_output=True, text=True).stdout)
    assert abs(float(probe["format"]["duration"]) - total/FS) < .1
    assert int(probe["format"]["size"]) == output_path.stat().st_size

    # Fully decode each final MP3, so metadata alone cannot hide truncation.
    decoded = subprocess.Popen([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(output_path),
                                "-f", "f32le", "-ar", str(FS), "-ac", "2", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    decoded_samples = 0
    peak = 0.0
    sumsq = 0.0
    while True:
        chunk = decoded.stdout.read(8 * 131072)
        if not chunk:
            break
        values = np.frombuffer(chunk, dtype="<f4")
        assert len(values) % 2 == 0 and np.all(np.isfinite(values))
        decoded_samples += len(values)//2
        peak = max(peak, float(np.max(abs(values))))
        sumsq += float(np.sum(values.astype(float)**2))
    error = decoded.stderr.read().decode("utf-8", errors="replace")
    assert decoded.wait() == 0, error
    assert decoded_samples == total
    assert peak < .95
    waveform_bins = 1600
    block = max(1, total // waveform_bins)
    peaks = np.max(abs(mix[:block*waveform_bins]).reshape(waveform_bins, block, 2), axis=(1, 2))
    report = {
        "file": output_path.name, "title": title, "composer": composer,
        "source_midi": Path(midi_path).name, "source_sha256": hashlib.sha256(Path(midi_path).read_bytes()).hexdigest(),
        "sample_rate_hz": FS, "channels": 2, "bitrate": 192000,
        "duration_s": total/FS, "original_midi_duration_s": midi["duration"],
        "note_count": rendered, "max_midi_polyphony": polyphony(midi["notes"]),
        "pedal_extended_notes": sum(n.end > n.key_end+.01 for n in midi["notes"]),
        "midi_channels_used": sorted(ch+1 for ch in groups), "tempo_events": len(midi["tempo_events"]),
        "unmatched_note_offs": midi["unmatched_note_offs"], "forced_note_ends": midi["forced_note_ends"],
        "programs_zero_based": midi["programs"], "track_names": midi["track_names"],
        "source_text_and_credits": midi["texts"], "master_gain": master_gain,
        "decoded_samples_per_channel": decoded_samples, "decoded_peak": peak,
        "decoded_rms": math.sqrt(sumsq/(2*decoded_samples)), "file_size_bytes": output_path.stat().st_size,
        "mp3_probe": probe, "render_wall_seconds": time.monotonic()-began,
        "models": list(used_models.values()), "waveform_peaks": peaks.tolist(),
    }
    (data_dir / (output_path.stem+"_report.json")).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (data_dir / (output_path.stem+"_notes.json")).write_text(json.dumps([asdict(n) for n in midi["notes"]], ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("file", "duration_s", "note_count", "decoded_peak", "file_size_bytes", "render_wall_seconds")}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("midi", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--composer")
    parser.add_argument("--preview-seconds", type=float)
    args = parser.parse_args()
    output = args.output or args.midi.with_name(args.midi.stem+"_pratt.mp3")
    render_midi(args.midi, output, args.title, args.composer, args.preview_seconds)


if __name__ == "__main__":
    main()
