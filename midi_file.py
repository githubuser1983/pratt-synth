"""Small dependency-free Standard MIDI File reader for formats 0 and 1.

Handles PPQ/SMPTE timing, running status, tempo, channel controls, sustain,
programs, pitch bends (including RPN bend range), and overlapping notes.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
import struct


@dataclass
class Note:
    start: float
    end: float
    key_end: float
    pitch: int
    velocity: int
    program: int
    channel: int
    track: int


def vlq(data, offset):
    value = 0
    for _ in range(4):
        b = data[offset]
        offset += 1
        value = (value << 7) | (b & 127)
        if b < 128:
            return value, offset
    raise ValueError("Invalid MIDI variable-length integer")


def read_midi(path):
    data = Path(path).read_bytes()
    if data[:4] != b"MThd":
        raise ValueError("Expected a Standard MIDI File")
    length = struct.unpack_from(">I", data, 4)[0]
    fmt, track_count, division = struct.unpack_from(">HHH", data, 8)
    if fmt not in (0, 1):
        raise ValueError("MIDI format 2 contains independent sequences; use format 0 or 1")
    position = 8 + length
    raw = []
    track_names = {}
    track_ends = []
    serial = 0
    texts = []
    for track in range(track_count):
        if data[position:position+4] != b"MTrk":
            raise ValueError("Expected MTrk chunk")
        size = struct.unpack_from(">I", data, position + 4)[0]
        chunk = data[position+8:position+8+size]
        position += 8 + size
        offset = tick = 0
        running = None
        while offset < len(chunk):
            delta, offset = vlq(chunk, offset)
            tick += delta
            status = chunk[offset]
            if status & 128:
                offset += 1
            else:
                if running is None:
                    raise ValueError("Missing running status")
                status = running
            if status == 255:
                kind = chunk[offset]
                offset += 1
                count, offset = vlq(chunk, offset)
                payload = chunk[offset:offset+count]
                offset += count
                if kind == 81:
                    if count != 3:
                        raise ValueError("Invalid tempo event")
                    raw.append((tick, serial, track, "tempo", int.from_bytes(payload, "big")))
                elif kind == 3:
                    track_names[track] = payload.decode("latin1", errors="replace").strip()
                elif kind in (1, 2):
                    texts.append(payload.decode("latin1", errors="replace"))
                serial += 1
                if kind == 47:
                    break
                continue
            if status in (240, 247):
                count, offset = vlq(chunk, offset)
                offset += count
                running = None
                continue
            if status < 128 or status >= 240:
                raise ValueError(f"Unsupported status {status:#x}")
            running = status
            kind, channel = status & 240, status & 15
            count = 1 if kind in (192, 208) else 2
            payload = tuple(chunk[offset:offset+count])
            if len(payload) != count or any(x > 127 for x in payload):
                raise ValueError("Invalid channel message")
            offset += count
            raw.append((tick, serial, track, kind, (channel, *payload)))
            serial += 1
        track_ends.append(tick)
    raw.sort(key=lambda e: (e[0], e[1]))
    end_tick = max(track_ends, default=0)
    seconds = 0.0
    last_tick = 0
    tempo = 500000
    smpte_rate = None
    if division & 0x8000:
        fps_byte = (division >> 8) - 256
        fps = 29.97 if fps_byte == -29 else -fps_byte
        smpte_rate = fps * (division & 255)
        if smpte_rate <= 0:
            raise ValueError("Invalid SMPTE division")
    programs = [0] * 16
    pedals = [False] * 16
    active = defaultdict(deque)
    held = defaultdict(list)
    notes = []
    controls = {c: {7: [(0.0, 100)], 10: [(0.0, 64)], 11: [(0.0, 127)],
                    1: [(0.0, 0)], "bend": [(0.0, 0.0)]} for c in range(16)}
    bend_value = [8192] * 16
    bend_range = [2.0] * 16
    rpn = [[127, 127] for _ in range(16)]
    tempo_events = []
    counts = defaultdict(int)
    channel_programs = defaultdict(set)
    note_on_count = 0
    unmatched_offs = 0

    def finish(note, end, key_end=None):
        note.end = max(end, note.start + 0.002)
        if key_end is not None:
            note.key_end = key_end
        notes.append(note)

    def release_held(channel, time):
        for note in held.pop(channel, []):
            finish(note, time)

    def end_channel(channel, time, force):
        for (tr, ch, key), queue in list(active.items()):
            if ch == channel:
                while queue:
                    note = queue.popleft()
                    note.key_end = time
                    if pedals[ch] and not force:
                        held[ch].append(note)
                    else:
                        finish(note, time)
        if force:
            release_held(channel, time)

    for tick, order, track, kind, payload in raw:
        seconds += (tick-last_tick) / smpte_rate if smpte_rate else (tick-last_tick) * tempo / (1e6 * division)
        last_tick = tick
        if kind == "tempo":
            tempo = payload
            tempo_events.append((seconds, tempo))
            continue
        channel = payload[0]
        counts[str(kind)] += 1
        if kind == 192:
            programs[channel] = payload[1]
            channel_programs[channel].add(payload[1])
        elif kind == 144 and payload[2] > 0:
            note_on_count += 1
            note = Note(seconds, seconds, seconds, payload[1], payload[2],
                        programs[channel], channel, track)
            active[(track, channel, payload[1])].append(note)
        elif kind == 128 or (kind == 144 and payload[2] == 0):
            key = (track, channel, payload[1])
            if not active[key]:
                unmatched_offs += 1
                continue
            note = active[key].popleft()
            note.key_end = seconds
            if pedals[channel]:
                held[channel].append(note)
            else:
                finish(note, seconds)
        elif kind == 176:
            cc, value = payload[1:]
            if cc in (1, 7, 10, 11):
                controls[channel][cc].append((seconds, value))
            elif cc == 64:
                on = value >= 64
                if pedals[channel] and not on:
                    release_held(channel, seconds)
                pedals[channel] = on
            elif cc in (101, 100):
                rpn[channel][0 if cc == 101 else 1] = value
            elif cc == 6 and rpn[channel] == [0, 0]:
                bend_range[channel] = float(value)
                bend = (bend_value[channel]-8192) / 8192 * bend_range[channel]
                controls[channel]["bend"].append((seconds, bend))
            elif cc == 120:
                end_channel(channel, seconds, True)
            elif cc == 123:
                end_channel(channel, seconds, False)
            elif cc == 121:
                release_held(channel, seconds)
                pedals[channel] = False
                controls[channel][11].append((seconds, 127))
                controls[channel][1].append((seconds, 0))
                controls[channel]["bend"].append((seconds, 0.0))
                bend_value[channel] = 8192
        elif kind == 224:
            bend_value[channel] = payload[1] | (payload[2] << 7)
            bend = (bend_value[channel]-8192) / 8192 * bend_range[channel]
            controls[channel]["bend"].append((seconds, bend))
    seconds += (end_tick-last_tick) / smpte_rate if smpte_rate else (end_tick-last_tick)*tempo/(1e6*division)
    forced_notes = 0
    for queue in active.values():
        while queue:
            note = queue.popleft()
            finish(note, seconds, seconds)
            forced_notes += 1
    for channel in list(held):
        release_held(channel, seconds)
    notes.sort(key=lambda n: (n.start, n.track, n.channel, n.pitch))
    assert len(notes) == note_on_count
    return {
        "path": str(Path(path).resolve()), "format": fmt, "tracks": track_count,
        "division": division, "duration": seconds, "notes": notes,
        "track_names": track_names, "texts": texts, "controls": controls,
        "tempo_events": tempo_events, "programs": {ch: sorted(p) for ch, p in channel_programs.items()},
        "unmatched_note_offs": unmatched_offs, "forced_note_ends": forced_notes,
        "event_counts": dict(counts),
    }


def polyphony(notes):
    points = [(n.start, 1) for n in notes] + [(n.end, -1) for n in notes]
    count = largest = 0
    for _, delta in sorted(points):
        count += delta
        largest = max(largest, count)
    return largest
