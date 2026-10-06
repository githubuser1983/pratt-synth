# Mehrstimmiger Pratt-MIDI-Synthesizer

Dieser eigenständige Offline-Synthesizer wandelt Standard-MIDI-Dateien in
Stereo-MP3 um. Die sechs beigefügten Stücke wurden vollständig gerendert.
Die Aufnahmen verwenden eigene mathematisch definierte Pratt-Klangfarben.

## Schnellstart

Voraussetzungen: Python 3.10 oder neuer, FFmpeg einschließlich ffprobe im
Suchpfad, NumPy und SciPy. Matplotlib wird für optionale Übersichten verwendet.

```bash
python -m pip install -r source/requirements.txt
python source/pratt_midi_synth.py "meine_datei.mid" --output "meine_aufnahme.mp3"
```

Für die sechs mitgelieferten Dateien:

```bash
python source/render_all.py --workers 2
```

Zum schnellen Test einer fremden MIDI-Datei:

```bash
python source/pratt_midi_synth.py "meine_datei.mid" --output "hoerprobe.mp3" --preview-seconds 30
```

Die Presets stehen als gut lesbares Dictionary `PRESETS` in
`source/pratt_midi_synth.py`. Dort können Pratt-Basiszahl, Obertonverlauf,
Filterfrequenzskala, Anschwingzeit, Ausklingzeit und Vibrato verändert werden.

## 1. Pratt-Polynome

Die Grundlage ist dieselbe rekursive Familie wie in den bisherigen Versuchen:

\[
f_1(x)=1,\qquad f_2(x)=x,
\]

\[
f_p(x)=1+f_{p-1}(x)\quad(p\text{ ungerade prim}),\qquad
f_n(x)=\prod_{p\mid n}f_p(x)^{v_p(n)}.
\]

Alle Polynomkoeffizienten werden exakt als ganze Zahlen berechnet.
Es gelten \(f_n(2)=n\) und \(f_{mn}=f_mf_n\).

Der stabile analoge Filter aus den vorherigen Diagrammen ist

\[
H_n(s)=\frac{n}{f_n(2+s/\omega_0)}.
\]

Für seine Nullstellen \(\theta_j\) kann man dieselbe Antwort schreiben als

\[
H_n(s)=\prod_j\frac{2-\theta_j}{2+s/\omega_0-\theta_j}.
\]

Die Formung der Obertöne kommt damit vollständig aus der Pratt-Polynom-
beziehungsweise Nullstellenstruktur. Die Kaskadenidentität gilt weiter:
\(H_{mn}=H_mH_n\).

## 2. Von MIDI-Tonhöhen zu Pratt-Klangfarben

Für MIDI-Taste m ist die musikalische Grundfrequenz

\[
f_0(m)=440\cdot2^{(m-69)/12}\ \mathrm{Hz}.
\]

Die Note muss diese Tonhöhe behalten, damit die Komposition erhalten bleibt.
Deshalb werden harmonische Obertöne \(k f_0\) erzeugt und durch den Pratt-
Filter gewichtet. Dies ist eine eigene Synthesizer-Fassung der Sonifikation,
mit musikalisch vorgegebenen Frequenzen.

Für Instrumentenfamilie g wählen wir eine Basiszahl \(b_g\) und setzen

\[
n_{m,g}=(m+1)b_g,\qquad
R_{m,g}(k)=\frac{n_{m,g}}{f_{n_{m,g}}(2+i\xi_g k)}.
\]

Hier wird \(\omega_0=2\pi f_0/\xi_g\) gewählt, also ist
\(R_{m,g}(k)=H_{n_{m,g}}(2\pi i kf_0)\).
Damit ist die Tonhöhenzuordnung musikalisch exakt, während der Zahlenindex
und die Instrumentenbasis die spektrale Form bestimmen.

Vor Pegelanpassung und Hüllkurve lautet die periodische Wellenform

\[
w_{m,g}(t)=\operatorname{Re}\sum_{k=1}^{K}
\left(-i\,a_{k,g}\,R_{m,g}(k)\right)e^{2\pi i kf_0 t}.
\]

Die reellen Anregungsgewichte \(a_{k,g}\) fallen ungefähr wie
\(k^{-\sigma_g}\); Flöte und Rohrblatt besitzen zusätzlich eine einfache
spektrale Gewichtung. Betrag und Phase von R werden beide verwendet.
Die stationäre Filterantwort wird in einer bandbegrenzten Wavetable gespeichert.
Die Synthese simuliert dadurch die stationären Obertöne des Filters; die
Anschwing- und Loslassvorgänge werden durch die dokumentierten Hüllkurven
gestaltet, statt jeden analogen Filtertransienten nachzurechnen.

| Familie | Pratt-Basiszahl | Charakter |
|---|---:|---|
| Klavier | 5 | rascher Anschlag, natürliche Abnahme, allmählich dunkler |
| E-Piano | 11 | weicher Anschlag und längere Abnahme |
| Orgel | 2 | gehaltene Töne, kurzer Release |
| Cembalo / Zupfen / Pizzicato | 17 | kurzer, obertonreicher Anschlag |
| Bass | 3 | tiefe, weich abklingende Stimmen |
| Streicher | 7 | gehaltene Stimmen mit leichtem Vibrato |
| Blechbläser | 11 | kräftigeres Obertonspektrum |
| Rohrblatt | 5 | stärker gewichtete ungerade Obertöne |
| Flöte | 3 | wenige kräftige Obertöne |
| Synth-Streicher / Pad | 37 | sanfterer Aufbau und längerer Release |
| Pauken | 7 | rascher, tief abklingender Anschlag |

Die General-MIDI-Programmnummer entscheidet über diese Familie. Änderungen
innerhalb derselben Familie können denselben mathematischen Grundklang
verwenden. Die konkrete MIDI-Taste verändert zusätzlich den Pratt-Zahlenindex.
Für MIDI-Kanal 10 gibt es einfache prozedurale Percussion-Presets.

## 3. Polyphonie und MIDI-Steuerung

Jede MIDI-Note hat eine eigene Wellenformphase und Hüllkurve. Stimmen werden
überlagert; der Offline-Renderer hat keine feste Grenze für die Stimmenzahl.
Die geprüften Dateien benötigen bis zu 100 gleichzeitig durch das Pedal
gehaltene Noten. Freie Klavier- und Zupfklänge werden erst unterhalb ihrer
sehr kleinen natürlichen Restamplitude begrenzt; es gibt kein Voice-Stealing.

Der eingebaute MIDI-Leser behandelt:

- MIDI-Format 0 und 1, mehrere Tracks und Running Status;
- Tempoereignisse und PPQ- beziehungsweise SMPTE-Zeitbasis;
- Note-On, Note-Off und Note-On mit Anschlagstärke 0;
- wiederholte und überlappende Noten gleicher Tonhöhe;
- Sustain-Pedal, Programmwechsel, Kanal-Lautstärke, Expression und Panorama;
- Pitch Bend mit Standardbereich und einfachem RPN-Bend-Range;
- All Notes Off und All Sound Off.

SysEx-Befehle und weitere Controller bleiben ohne Synthese-Wirkung.
MIDI-Format 2 mit unabhängigen Sequenzen wird mit einer klaren Fehlermeldung
abgewiesen. Der Tonumfang wird nach oben bandbegrenzt; die berücksichtigten
Obertöne liegen mit Abstand unter der Nyquist-Frequenz.

Anschlagstärke steuert Pegel und Helligkeit. Lautstärke, Expression und
Panorama werden auch während gehaltener Noten ausgewertet. Kurze Rampen
vermeiden Sprünge bei Controlleränderungen. Ein gemeinsamer Stereo-Raumklang
und eine lineare Gesamtpegelanpassung vervollständigen die Aufnahme.
Hüllkurven, Panorama und Pegelanpassung sind Präsentationsschritte; für die
fertigen Musikdateien wird kein exaktes arithmetisches Kaskadengesetz behauptet.

## 4. Die sechs Aufnahmen

Die originalen MIDI-Dateien liegen in `midis/`. Ihre Noten und Zeitachsen
werden nicht verändert. Track-Namen, Quellenangaben und vorhandene
Urhebervermerke werden in den Berichten unter `data/` aufbewahrt.

Die Ausgabe verwendet 44.1 kHz, Stereo und MP3 mit 192 kbit/s.
Jedes Stück enthält zusätzlich 3.2 Sekunden Raum- und Ausklangzeit.
Die Variante `vivaldi_estate_3_D4.mid` wird als eigene Performance gerendert;
ihre im MIDI vorhandenen Unterschiede werden übernommen.

## 5. Prüfung und Reproduzierbarkeit

```bash
python source/validate_synth.py
```

Der gezielte Test prüft Tempoänderung, Running Status und zwei durch das
Pedal gleichzeitig verlängerte Noten anhand bekannter Zeitpunkte. Er prüft
außerdem die Pratt-Kaskadenidentität sowie die Tonhöhe A4 bei 440 Hz.

Alle sechs Aufnahmen wurden anschließend vollständig aus ihrer MP3-Datei
zurückdekodiert. Sampleanzahl, Laufzeit, endliche Samples und Spitzenpegel
werden kontrolliert. Die Zahl gerenderter Noten muss der MIDI-Eingabe
entsprechen. JSON-Berichte enthalten Noten, Programme, verwendete
Polynome, Tempoanzahl, Pegel und eine SHA-256-Prüfsumme der MIDI-Quelle.

Die Ergebnisse der gesamten Sammlung stehen in
`data/alle_stuecke_verifiziert.json`.
