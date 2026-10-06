#!/usr/bin/env python3
"""Focused checks for the new tempo/pedal reader and Pratt sound model."""
from pathlib import Path
import json
import struct
import tempfile
import numpy as np

from midi_file import Note, read_midi, polyphony
from pratt_midi_synth import FS, melodic_voice, response, multiply, pratt


def main():
    # PPQ=100: tempo changes after half a second, then the pedal releases
    # two keys together at 1.5 s. One note-on uses running status.
    data = (b"\x00\xff\x51\x03\x07\xa1\x20"
            b"\x00\xb0\x40\x7f"
            b"\x00\x90\x3c\x64"
            b"\x64\x90\x3c\x00"
            b"\x00\xff\x51\x03\x0f\x42\x40"
            b"\x19\x40\x64"
            b"\x19\x80\x40\x00"
            b"\x32\xb0\x40\x00"
            b"\x19\xff\x2f\x00")
    payload = b"MThd"+struct.pack(">IHHH", 6, 0, 1, 100)+b"MTrk"+struct.pack(">I", len(data))+data
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/"timing_and_pedal.mid"
        path.write_bytes(payload)
        midi = read_midi(path)
    assert len(midi["notes"]) == 2
    a, b = midi["notes"]
    assert (a.start, a.key_end, a.end) == (0.0, .5, 1.5)
    assert (b.start, b.key_end, b.end) == (.75, 1.0, 1.5)
    assert midi["duration"] == 1.75 and polyphony(midi["notes"]) == 2
    assert midi["unmatched_note_offs"] == 0
    x = np.linspace(.01, 20, 3000)
    assert pratt(35) == multiply(pratt(5), pratt(7))
    error = np.max(abs(response(35, x) - response(5, x)*response(7, x)))
    assert error < 1e-12
    note = Note(0, .5, .5, 69, 100, 0, 0, 0)
    wave, n, h = melodic_voice(note, {"bend": [(0, 0)]}, 2)
    spectrum = abs(np.fft.rfft(wave*np.hanning(len(wave))))
    frequency = np.fft.rfftfreq(len(wave), 1/FS)[np.argmax(spectrum)]
    assert abs(frequency-440) < 2
    assert np.all(np.isfinite(wave)) and h*440 < FS*.5
    print(json.dumps({"tempo_sustain_and_running_status": "passed", "pratt_cascade_max_abs_error": float(error),
                      "midi_A4_spectral_peak_hz": float(frequency), "midi_A4_pratt_index": n,
                      "anti_alias_harmonic_limit": h}, indent=2))


if __name__ == "__main__":
    main()
