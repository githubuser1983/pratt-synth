#!/usr/bin/env python3
"""Render the six attached MIDI files with two independent audio workers."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json

from pratt_midi_synth import render_midi

JOBS = [
    ("vivaldi_4_stagioni_estate_3_(c)pollen.mid", "vivaldi_sommer_3_pratt.mp3", "Vivaldi – Sommer, III · Pratt", "Antonio Vivaldi"),
    ("vivaldi_4_stagioni_primavera_3_(c)pollen.mid", "vivaldi_fruehling_3_pratt.mp3", "Vivaldi – Frühling, III · Pratt", "Antonio Vivaldi"),
    ("gymnopedie_1_(c)oguri.mid", "satie_gymnopedie_1_pratt.mp3", "Satie – Gymnopédie n° 1 · Pratt", "Erik Satie"),
    ("6103d_moonlight_sonata_27-2_3_(nc)smythe.mid", "beethoven_mondschein_3_pratt.mp3", "Beethoven – Mondscheinsonate, III · Pratt", "Ludwig van Beethoven"),
    ("vivaldi_estate_3_D4.mid", "vivaldi_sommer_3_D4_pratt.mp3", "Vivaldi – Sommer, III, D4-Datei · Pratt", "Antonio Vivaldi"),
    ("beethoven_symphony_5_1_(c)galimberti.mid", "beethoven_sinfonie_5_1_pratt.mp3", "Beethoven – Sinfonie Nr. 5, I · Pratt", "Ludwig van Beethoven"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--midi-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--reuse-verified", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    midi_dir = args.midi_dir or project/"midis"
    audio_dir = args.output_dir or project/"audio"
    reports = {}
    tasks = []
    for source, filename, title, composer in JOBS:
        output = audio_dir/filename
        report = audio_dir.parent/"data"/(output.stem+"_report.json")
        if args.reuse_verified and output.exists() and report.exists():
            item = json.loads(report.read_text())
            if output.stat().st_size == item["file_size_bytes"]:
                reports[filename] = item
                continue
        tasks.append((midi_dir/source, output, title, composer))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_midi, *args): args[1].name for args in tasks}
        for future in as_completed(futures):
            report = future.result()
            reports[report["file"]] = report
            print("COMPLETE:", report["file"], flush=True)
    ordered = [reports[job[1]] for job in JOBS]
    summary = [{k: r[k] for k in ("file", "title", "duration_s", "note_count", "max_midi_polyphony",
                                   "pedal_extended_notes", "tempo_events", "decoded_peak", "decoded_rms",
                                   "file_size_bytes", "render_wall_seconds")} for r in ordered]
    project.joinpath("data", "alle_stuecke_verifiziert.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
