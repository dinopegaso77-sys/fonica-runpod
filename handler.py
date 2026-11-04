"""Fonica Infinity - worker RunPod: separa un brano con Demucs e trascrive il testo con Whisper.

Due modi d'uso:

1) Separazione (con trascrizione facoltativa)
   {"input": {"audio_b64": "...", "filename": "brano.mp3",
              "model": "htdemucs" | "htdemucs_6s" | "htdemucs_ft", "bitrate": 128,
              "transcribe": true, "language": "it" | "" , "prompt": "testo noto (facoltativo)"}}
   -> {"stems": {"vocals": "<m4a base64>", ...}, "format": "m4a", "bitrate": 128,
       "lyrics": {...} | null, "seconds": 41.2}

2) Solo trascrizione (la voce è già separata)
   {"input": {"mode": "transcribe", "audio_b64": "<wav 16 kHz mono>", "language": "", "prompt": ""}}
   -> {"lyrics": {...}, "seconds": 12.3}

"lyrics" = {"language": "it", "segments": [{"s": 12.4, "e": 15.9, "text": "...",
             "words": [{"w": "parola", "s": 12.4, "e": 12.8}, ...]}, ...]}
3) Voce da studio (dopo una registrazione)
   {"input": {"mode": "enhance", "audio_b64": "<voce registrata>", "filename": "voce.webm"}}
   -> {"voice": "<flac mono base64>", "format": "flac", "seconds": 9.1}

In caso di problemi: {"error": "messaggio"}
"""
import base64
import glob
import os
import re
import shutil
import subprocess
import tempfile
import time

import runpod

MODELS = {"htdemucs", "htdemucs_6s", "htdemucs_ft"}
LIMIT = 18_500_000  # caratteri base64 in uscita: RunPod accetta circa 20 MB per risposta
ASR_MODEL = os.environ.get("WHISPER_MODEL", "turbo")

# frasi che Whisper "inventa" sui silenzi (titoli di coda dei sottotitoli e simili)
HALLU = re.compile(
    r"sottotitoli|a cura di|qtss|amara\.org|grazie per la visione|iscriviti|"
    r"thank(s)? (you )?for watching|subscribe|subtitles by|untertitel|sous-titres",
    re.I,
)

_asr = None


def asr_model():
    global _asr
    if _asr is None:
        import torch
        import whisper
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _asr = whisper.load_model(ASR_MODEL, device=dev)
    return _asr


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "errore sconosciuto").strip()
        print("COMANDO FALLITO:", " ".join(cmd[:4]), "\n", err[-6000:], flush=True)  # nei Logs di RunPod
        lines = [l.strip() for l in err.splitlines() if l.strip()]
        raise RuntimeError(" | ".join(lines[-4:])[-700:])
    return p


def gpu_name():
    try:
        import torch
        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "nessuna"
    except Exception as e:  # solo informativo
        return "sconosciuta (" + str(e)[:80] + ")"


def transcribe(path, language=None, prompt=None):
    import torch
    m = asr_model()
    res = m.transcribe(
        path,
        language=language or None,
        task="transcribe",
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt=(prompt or None),
        fp16=torch.cuda.is_available(),
        verbose=None,
    )
    segs = []
    for sg in res.get("segments", []):
        text = (sg.get("text") or "").strip()
        if not text or HALLU.search(text):
            continue
        if sg.get("no_speech_prob", 0) > 0.6 and sg.get("avg_logprob", 0) < -0.8:
            continue
        words = []
        for w in sg.get("words") or []:
            t = (w.get("word") or "").strip()
            if t:
                words.append({"w": t, "s": round(float(w["start"]), 2), "e": round(float(w["end"]), 2)})
        if words:
            segs.append({"s": words[0]["s"], "e": words[-1]["e"], "text": text, "words": words})
    return {"language": res.get("language"), "segments": segs}


def handler(job):
    inp = job.get("input") or {}
    data = inp.get("audio_b64")
    if not data:
        return {"error": "manca il file audio"}
    lang = (inp.get("language") or "").strip() or None
    prompt = (inp.get("prompt") or "").strip()[:800] or None
    t0 = time.time()
    work = tempfile.mkdtemp()
    try:
        name = os.path.basename(inp.get("filename") or "brano.mp3")
        ext = os.path.splitext(name)[1].lower() or ".mp3"
        src = os.path.join(work, "input" + ext)
        with open(src, "wb") as f:
            f.write(base64.b64decode(data))

        # --- solo trascrizione ---
        if inp.get("mode") == "transcribe":
            runpod.serverless.progress_update(job, "Trascrivo il testo dalla voce")
            return {"lyrics": transcribe(src, lang, prompt), "seconds": round(time.time() - t0, 1)}

        # --- voce da studio: toglie la base entrata nel microfono, il rumore e le «s» forti ---
        if inp.get("mode") == "enhance":
            gpu = gpu_name()
            runpod.serverless.progress_update(job, "Pulisco la voce sulla scheda video")
            wav = os.path.join(work, "voce.wav")
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "2", "-ar", "44100", wav])
            out = os.path.join(work, "out")
            base = ["python", "-m", "demucs", "-n", "htdemucs", "--two-stems", "vocals", "-o", out, "--filename", "{stem}.{ext}"]
            try:
                run(base + ["-d", "cuda" if gpu != "nessuna" else "cpu", wav])
            except RuntimeError:
                shutil.rmtree(out, ignore_errors=True)
                run(base + ["-d", "cpu", wav])
            voc = os.path.join(out, "htdemucs", "vocals.wav")
            if not os.path.exists(voc):
                return {"error": "Demucs non ha prodotto la voce"}
            runpod.serverless.progress_update(job, "Rifinisco la voce")
            flac = os.path.join(work, "voce.flac")
            # FLAC mono: nessun ritardo aggiunto (la voce resta a tempo) e file piccolo
            enc = ["-ac", "1", "-ar", "44100", "-c:a", "flac", flac]
            try:
                run(["ffmpeg", "-y", "-loglevel", "error", "-i", voc, "-af", "highpass=f=70,afftdn=nf=-30,deesser"] + enc)
            except RuntimeError:  # ffmpeg senza questi filtri: basta la voce separata
                run(["ffmpeg", "-y", "-loglevel", "error", "-i", voc] + enc)
            with open(flac, "rb") as fh:
                b = base64.b64encode(fh.read()).decode()
            if len(b) > LIMIT:
                return {"error": "voce troppo lunga per una sola richiesta"}
            return {"voice": b, "format": "flac", "gpu": gpu, "seconds": round(time.time() - t0, 1)}

        # --- separazione ---
        model = inp.get("model") or "htdemucs"
        if model not in MODELS:
            model = "htdemucs"
        gpu = gpu_name()
        print("Scheda video:", gpu, "- modello:", model, flush=True)
        runpod.serverless.progress_update(job, "Separo le tracce sulla scheda video")
        out = os.path.join(work, "out")
        base = ["python", "-m", "demucs", "-n", model, "-o", out, "--filename", "{stem}.{ext}"]
        note = None
        try:
            run(base + ["-d", "cuda" if gpu != "nessuna" else "cpu", src])
        except RuntimeError as e:
            # se la scheda video dà problemi, riprova sul processore: più lento ma funziona
            note = "scheda video %s non utilizzabile (%s): separato sul processore" % (gpu, str(e)[-200:])
            print(note, flush=True)
            runpod.serverless.progress_update(job, "La scheda video ha dato un errore: riprovo sul processore (più lento)")
            shutil.rmtree(out, ignore_errors=True)
            run(base + ["-d", "cpu", "-j", str(max(1, (os.cpu_count() or 2) - 1)), src])
        wavs = sorted(glob.glob(os.path.join(out, model, "*.wav")))
        if not wavs:
            return {"error": "Demucs non ha prodotto tracce"}

        lyrics = None
        if inp.get("transcribe"):
            voc = os.path.join(out, model, "vocals.wav")
            if os.path.exists(voc):
                runpod.serverless.progress_update(job, "Trascrivo il testo dalla voce")
                try:
                    lyrics = transcribe(voc, lang, prompt)
                except Exception as e:  # la separazione resta valida anche se la trascrizione fallisce
                    lyrics = {"error": str(e)[-300:], "segments": []}

        # tracce praticamente mute (per esempio il piano in un brano senza piano): non si inviano
        def silent(path):
            try:
                import soundfile as sf
                import numpy as np
                d, _ = sf.read(path, dtype="float32")
                return float(np.sqrt(np.mean(d * d))) < 0.003
            except Exception:
                return False
        wavs = [w for w in wavs if os.path.basename(w).startswith(("vocals", "bass", "drums", "other")) or not silent(w)]
        runpod.serverless.progress_update(job, "Comprimo le tracce")
        first = int(inp.get("bitrate") or 128)
        # qualità adattata alla durata: le tracce più importanti (voce, basso, chitarra) hanno più qualità
        try:
            import soundfile as sf
            dur = max(1.0, sf.info(wavs[0]).duration)
        except Exception:
            dur = 300.0
        weight = {"vocals": 1.3, "bass": 1.2, "guitar": 1.15, "drums": .9, "piano": .8, "other": .8}
        names = [os.path.splitext(os.path.basename(w))[0] for w in wavs]
        wsum = sum(weight.get(k, .8) for k in names) or 1
        total_kbps = LIMIT * .75 / (dur * 125) * .96  # caratteri base64 -> byte -> kbps disponibili
        scale = 1.0
        for _ in range(6):
            res, total, brs = {}, 0, {}
            for w, k in zip(wavs, names):
                br = int(max(32, min(first, total_kbps * weight.get(k, .8) / wsum * scale)))
                brs[k] = br
                m = os.path.join(work, f"{k}.m4a")
                run(["ffmpeg", "-y", "-loglevel", "error", "-i", w, "-c:a", "aac", "-b:a", f"{br}k", m])
                with open(m, "rb") as fh:
                    b = base64.b64encode(fh.read()).decode()
                res[k] = b
                total += len(b)
            if total < LIMIT:
                break
            scale *= .85 * LIMIT / total
        else:
            return {"error": "brano troppo lungo per una sola richiesta (oltre 10 minuti circa): prova un file più corto"}
        br = max(brs.values())
        return {"stems": res, "format": "m4a", "bitrate": br, "model": model, "gpu": gpu, "note": note,
                "lyrics": lyrics, "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # il messaggio torna all'app
        print("ERRORE:", repr(e)[-3000:], flush=True)
        return {"error": str(e)[-700:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
