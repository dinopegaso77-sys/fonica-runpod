"""
Pure Voltage - testi delle canzoni con Qwen3-ASR (al posto di Whisper, che resta come riserva).

Gira nell'ambiente separato /opt/qwen (librerie più nuove), lanciato da handler.py:
    /opt/qwen/bin/python /testi.py richiesta.json risposta.json
richiesta: {"audio": "voce.wav", "language": "it" | "" }
risposta:  {"language": "it", "segments": [{s, e, text, words: [{w, s, e}]}], "engine": ..., "models": ...}
          (stesso formato di Whisper: l'app non cambia)

I modelli NON sono nell'immagine: vengono dalla cartella Hugging Face di Pure Voltage.
  1) se l'endpoint ha la cartella nella cache di RunPod (campo «Model»), si usano da lì: avvio rapido, download gratis;
  2) altrimenti si scaricano una volta su questo worker (circa un minuto) e restano finché il worker è acceso.
Qualsiasi errore fa uscire con codice diverso da 0: handler.py usa Whisper come sempre.
"""
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.environ.get("PV_MODELS", "Purevoltage/pure-voltage-models")
ASR_SUB = "qwen3-asr-1.7b"
AL_SUB = "qwen3-forcedaligner-0.6b"
LOCAL = "/tmp/pvmodels"
SR = 16000

# lingue dell'allineatore (servono per i tempi parola per parola)
LANGS = {"it": "Italian", "en": "English", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese",
         "ru": "Russian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "yue": "Cantonese"}
CODES = {v.lower(): k for k, v in LANGS.items()}


def say(msg):
    """Avviso per l'app (handler.py lo inoltra come avanzamento)."""
    print("@@ " + msg, flush=True)


def ready(d):
    return all(os.path.exists(os.path.join(d, s, "config.json")) for s in (ASR_SUB, AL_SUB))


def models_dir():
    root = "/runpod-volume/huggingface-cache/hub/models--" + REPO.replace("/", "--") + "/snapshots"
    for d in sorted(glob.glob(root + "/*"), key=os.path.getmtime, reverse=True):
        if ready(d):
            return d, "cache"
    if ready(LOCAL):
        return LOCAL, "disco"
    say("Prima volta su questo worker: scarico il programma dei testi (circa un minuto)")
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, local_dir=LOCAL, allow_patterns=[ASR_SUB + "/*", AL_SUB + "/*"])
    if not ready(LOCAL):
        raise RuntimeError("la cartella %s non contiene ancora i modelli dei testi" % REPO)
    return LOCAL, "scaricato"


def load_audio(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True, timeout=300).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def pieces(x, lo=120, hi=170):
    """Taglia la voce in pezzi da 2-3 minuti nel punto più silenzioso (l'allineatore regge fino a 5 minuti)."""
    n, out, s, hop = len(x), [], 0, SR // 20
    while n - s > hi * SR:
        a, b = s + lo * SR, s + hi * SR
        seg = x[a:b]
        f = len(seg) // hop
        e = np.sqrt((seg[:f * hop].reshape(f, hop) ** 2).mean(1) + 1e-12)
        e = np.convolve(e, np.ones(8) / 8, "same")
        cut = a + int(np.argmin(e)) * hop
        out.append((s, cut))
        s = cut
    out.append((s, n))
    return out


def items(ts):
    """Le parole con i tempi, comunque la libreria le restituisca."""
    if ts is None:
        return []
    for attr in ("items", "words", "segments"):
        v = getattr(ts, attr, None)
        if isinstance(v, (list, tuple)):
            ts = v
            break
    if isinstance(ts, (list, tuple)) and len(ts) == 1 and isinstance(ts[0], (list, tuple)):
        ts = ts[0]
    return list(ts) if isinstance(ts, (list, tuple)) else []


def field(it, *names):
    for n in names:
        v = it.get(n) if isinstance(it, dict) else getattr(it, n, None)
        if v is not None:
            return v
    return None


def lines(words):
    """Raggruppa le parole in righe: si va a capo sulle pause o quando la riga è lunga."""
    segs, cur = [], []

    def flush():
        if cur:
            segs.append({"s": cur[0]["s"], "e": cur[-1]["e"], "text": " ".join(w["w"] for w in cur), "words": list(cur)})
            cur.clear()

    for w in words:
        if cur:
            gap = w["s"] - cur[-1]["e"]
            if gap > 0.7 or (len(cur) >= 8 and gap > 0.25) or len(cur) >= 14 or w["e"] - cur[0]["s"] > 8:
                flush()
        cur.append(w)
    flush()
    for sg in segs:
        sg["text"] = sg["text"][:1].upper() + sg["text"][1:]
    return segs


def main(req_path, out_path):
    t0 = time.time()
    req = json.load(open(req_path))
    code = (req.get("language") or "").strip().lower()[:3] or None
    if code and code not in LANGS:
        raise RuntimeError("lingua %s non gestita dal programma dei testi" % code)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("serve la scheda video")
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    d, origin = models_dir()
    say("Trascrivo il testo dalla voce")
    from qwen_asr import Qwen3ASRModel
    model = Qwen3ASRModel.from_pretrained(
        os.path.join(d, ASR_SUB), dtype=dt, device_map="cuda:0", max_inference_batch_size=4, max_new_tokens=2048,
        forced_aligner=os.path.join(d, AL_SUB), forced_aligner_kwargs=dict(dtype=dt, device_map="cuda:0"))
    x = load_audio(req["audio"])
    parts = pieces(x)
    lang = LANGS[code] if code else None
    res = model.transcribe(audio=[(x[a:b], SR) for a, b in parts],
                           language=[lang] * len(parts) if lang else None, return_time_stamps=True)
    words, found = [], {}
    for (a, b), r in zip(parts, res):
        ln = str(field(r, "language") or "").strip().lower()
        if ln:
            found[ln] = found.get(ln, 0) + (b - a)
        its = items(field(r, "time_stamps"))
        raw = []
        for it in its:
            t = str(field(it, "text", "word") or "").strip()
            st, en = field(it, "start_time", "start"), field(it, "end_time", "end")
            if t and st is not None and en is not None:
                raw.append([t, float(st), float(en)])
        dur = (b - a) / SR
        if raw and max(w[2] for w in raw) > dur * 3 + 10:  # tempi in millisecondi
            for w in raw:
                w[1] /= 1000.0
                w[2] /= 1000.0
        off = a / SR
        for t, st, en in raw:
            st, en = st + off, max(en, st + 0.05) + off
            words.append({"w": t, "s": round(st, 2), "e": round(en, 2)})
        if (field(r, "text") or "").strip() and not raw:
            raise RuntimeError("testo senza tempi (lingua %s)" % (ln or "?"))
    words.sort(key=lambda w: w["s"])
    if not code and found:
        code = CODES.get(max(found, key=found.get), None)
    out = {"language": code, "segments": lines(words), "engine": "qwen3-asr-1.7b", "models": origin,
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
