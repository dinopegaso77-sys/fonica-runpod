"""
Pure Voltage - ritmo e batteria più precisi sulla scheda video.

Gira nell'ambiente separato /opt/qwen, lanciato da handler.py:
    /opt/qwen/bin/python /ritmo.py richiesta.json risposta.json
richiesta: {"mix": "brano.wav" (o null), "drums": "batteria.wav" (o null), "device": "cuda"}
risposta:  {"beats": [...secondi], "downbeats": [...], "bpm": 120, "meter": 4, "down": 0,
            "drums": [[secondi, "BD"|"SD"|"TT"|"HH"|"CY"], ...], "engines": {...}, "notes": [...]}

- Battiti e primo tempo della battuta: Beat This! (CPJKU, JKU Linz, licenza MIT), sul brano intero.
- Batteria: ADTOF (Zehren, Alunno, Bientinesi; versione PyTorch di Xavier Riley), sulla traccia della batteria.
  Pesi ADTOF con licenza CC BY-NC-SA 4.0 (uso non commerciale).
Se una delle due parti non va, l'altra si consegna lo stesso; se non va nessuna, esce con errore e
l'app continua con il metodo che usa già sul dispositivo.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile


def say(msg):
    print("@@ " + msg, flush=True)


def beats_part(mix, dev):
    from beat_this.inference import File2Beats
    say("Trovo battiti e battute (Beat This!)")
    f2b = File2Beats(checkpoint_path="final0", device=dev, dbn=False)
    b, d = f2b(mix)
    b = [round(float(x), 3) for x in b]
    d = [round(float(x), 3) for x in d]
    if len(b) < 8:
        raise RuntimeError("troppo pochi battiti")
    gaps = sorted(b[i + 1] - b[i] for i in range(len(b) - 1))
    bpm = 60.0 / gaps[len(gaps) // 2]
    # battiti per battuta: quanti battiti ci sono di solito tra un primo tempo e l'altro
    meter, down = 4, 0
    if len(d) >= 3:
        idx = []
        for t in d:
            j = min(range(len(b)), key=lambda k: abs(b[k] - t))
            if abs(b[j] - t) < 0.07:
                idx.append(j)
        if len(idx) >= 3:
            diffs = sorted(idx[i + 1] - idx[i] for i in range(len(idx) - 1))
            m = diffs[len(diffs) // 2]
            if 2 <= m <= 7:
                meter = m
            down = idx[0] % meter
    return {"beats": b, "downbeats": d, "bpm": round(bpm, 1), "meter": meter, "down": down}


LANE = {35: "BD", 36: "BD", 38: "SD", 40: "SD", 37: "SD", 41: "TT", 43: "TT", 45: "TT", 47: "TT", 48: "TT", 50: "TT",
        42: "HH", 44: "HH", 46: "HH", 49: "CY", 51: "CY", 52: "CY", 53: "CY", 55: "CY", 57: "CY", 59: "CY"}


def read_midi(path):
    out = []
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(path)
        for ins in pm.instruments:
            for n in ins.notes:
                if n.pitch in LANE:
                    out.append([round(float(n.start), 3), LANE[n.pitch], int(n.velocity)])
    except ImportError:
        import mido
        mf = mido.MidiFile(path)
        t = 0.0
        for msg in mf:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0 and msg.note in LANE:
                out.append([round(t, 3), LANE[msg.note], int(msg.velocity)])
    out.sort(key=lambda x: x[0])
    return out


def drums_part(drums, dev, work):
    say("Scrivo la batteria (ADTOF)")
    mid = os.path.join(work, "batteria.mid")
    exe = os.path.join(os.path.dirname(sys.executable), "adtof")
    err = ""
    if os.path.exists(exe):
        r = subprocess.run([exe, "--audio", drums, "--out", mid, "--device", dev], capture_output=True, text=True)
        err = (r.stdout + r.stderr)[-600:]
    if not os.path.exists(mid):
        from adtof_pytorch import transcribe_to_midi
        try:
            transcribe_to_midi(drums, mid, device=dev)
        except TypeError:
            transcribe_to_midi(drums, mid)
    if not os.path.exists(mid):
        raise RuntimeError("ADTOF non ha scritto il file MIDI " + err)
    return read_midi(mid)


def main(req_path, out_path):
    with open(req_path) as fh:
        req = json.load(fh)
    dev = req.get("device") or "cuda"
    try:
        import torch
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
    except Exception:
        dev = "cpu"
    work = tempfile.mkdtemp()
    res, notes, eng = {}, [], {}
    try:
        if req.get("mix"):
            try:
                res.update(beats_part(req["mix"], dev))
                eng["beats"] = "beat_this-final0"
            except Exception as e:
                notes.append("battiti: " + (str(e).strip().splitlines() or [""])[-1][-200:])
        if req.get("drums"):
            try:
                res["drums"] = drums_part(req["drums"], dev, work)
                eng["drums"] = "adtof-pytorch"
            except Exception as e:
                notes.append("batteria: " + (str(e).strip().splitlines() or [""])[-1][-200:])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if not eng:
        raise SystemExit("ritmo e batteria non riusciti: " + "; ".join(notes))
    res["engines"], res["notes"] = eng, notes
    with open(out_path, "w") as fh:
        json.dump(res, fh)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
