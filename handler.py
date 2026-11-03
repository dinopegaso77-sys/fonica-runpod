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
        raise RuntimeError((p.stderr or p.stdout or "errore sconosciuto")[-800:])
    return p


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

        # --- separazione ---
        model = inp.get("model") or "htdemucs"
        if model not in MODELS:
            model = "htdemucs"
        runpod.serverless.progress_update(job, "Separo le tracce sulla scheda video")
        out = os.path.join(work, "out")
        run(["python", "-m", "demucs", "-n", model, "-o", out, "--filename", "{stem}.{ext}", src])
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

        runpod.serverless.progress_update(job, "Comprimo le tracce")
        first = int(inp.get("bitrate") or 128)
        for br in (first, 96, 64):
            res, total = {}, 0
            for w in wavs:
                k = os.path.splitext(os.path.basename(w))[0]
                m = os.path.join(work, f"{k}.m4a")
                run(["ffmpeg", "-y", "-loglevel", "error", "-i", w, "-c:a", "aac", "-b:a", f"{br}k", m])
                with open(m, "rb") as fh:
                    b = base64.b64encode(fh.read()).decode()
                res[k] = b
                total += len(b)
            if total < LIMIT:
                break
        else:
            return {"error": "brano troppo lungo per una sola richiesta: prova un file più corto"}
        return {"stems": res, "format": "m4a", "bitrate": br, "model": model,
                "lyrics": lyrics, "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # il messaggio torna all'app
        return {"error": str(e)[-800:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
