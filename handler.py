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

4) Chi canta (duetti): riconosce le due voci nella traccia della voce
   {"input": {"mode": "voices", "audio_b64": "<voce separata>", "filename": "vocals.wav"}}
   -> {"singers": {"v": 2, "duet": true, "hop": 0.75, "win": 1.5, "k": "0011133222...", "sim": 0.41,
       "act": [indici delle finestre con voce], "emb": "<impronte int8, 'd' numeri per finestra attiva>", "qs": scala,
       "f0": [Hz], "poly": [0-100], "rms": [0-1000]}, "seconds": 6.2}
   "k": una cifra ogni "hop" secondi (finestra lunga "win"): 0 = silenzio, 1 = prima voce, 2 = seconda voce, 3 = insieme.
   L'app usa impronte, altezza e polifonia per decidere riga per riga e per ricalcolare dopo le correzioni a mano.
   Anche la separazione accetta "voices": true e allora risponde anche con "singers".

5) Tonalità pronte: sposta di semitoni le tracce già separate (nessuna nuova separazione)
   {"input": {"mode": "tone", "audio_url": "<traccia della chitarra o del basso, indirizzo firmato R2>", "role": "chitarra", "bpm": 130}}
   → suoni del brano (distorsione, delay, riverbero, curva delle frequenze) pronti per la pedaliera

   {"input": {"mode": "shift", "files": [{"name": "vocals.m4a", "b64": "..."}, ...], "semis": [-1, 1, -2]}}
   -> {"shift": {"-1": {"vocals": "<m4a base64>", ...}, "1": {...}}, "format": "m4a", "seconds": 20.4}
   La voce mantiene il suo timbro (formanti conservate); ogni traccia esce con la stessa qualità con cui è arrivata.

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
_spk = None

# soglie per riconoscere un duetto (somiglianza coseno fra le due voci medie)
DUET_SIM = float(os.environ.get("DUET_SIM", "0.72"))
BOTH_GAP = float(os.environ.get("BOTH_GAP", "0.06"))


def asr_model():
    global _asr
    if _asr is None:
        import torch
        import whisper
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _asr = whisper.load_model(ASR_MODEL, device=dev)
    return _asr


def spk_model():
    """Modello che riconosce le voci (SpeechBrain ECAPA): dà un'impronta di 192 numeri per ogni pezzo di voce."""
    global _spk
    if _spk is None:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _spk = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="/models/ecapa",
                                              run_opts={"device": dev})
    return _spk


def _yin(f, sr, fmin=70.0, fmax=1000.0):
    """Altezza (Hz) e aperiodicità (0 = una nota pulita, alta = più note o rumore) di un pezzetto di voce."""
    import numpy as np
    x = f.astype(np.float64) - float(f.mean())
    W = len(x) // 2
    lo, hi = int(sr / fmax), min(int(sr / fmin), W)
    N = 1 << int(np.ceil(np.log2(2 * len(x))))
    r = np.fft.irfft(np.conj(np.fft.rfft(x[:W], N)) * np.fft.rfft(x, N), N)[:hi + 2]
    cs = np.concatenate([[0.0], np.cumsum(x * x)])
    tau = np.arange(hi + 2)
    d = cs[W] + (cs[tau + W] - cs[tau]) - 2 * r
    d[0] = 0
    cm = np.cumsum(d[1:])
    dn = np.ones_like(d)
    dn[1:] = d[1:] * np.arange(1, len(d)) / np.maximum(cm, 1e-12)
    seg = dn[lo:hi + 1]
    cand = np.where((seg[1:-1] < 0.2) & (seg[1:-1] <= seg[:-2]) & (seg[1:-1] <= seg[2:]))[0]
    i = (cand[0] + 1) if len(cand) else int(np.argmin(seg))
    t = i + lo
    return sr / t, float(dn[t])


def pitch_feats(x, sr, starts, n):
    """Per ogni finestra: altezza della voce principale (Hz) e «polifonia» (0-1).
    Polifonia = quanta energia NON sta sulle armoniche della nota principale, insieme all'aperiodicità:
    una voce sola dà valori bassi, due voci con note diverse valori alti, un coro sotto valori medi."""
    import numpy as np
    fr, hp, nfft = 2048, 1024, 4096
    df = sr / nfft
    fq = np.arange(nfft // 2 + 1) * df
    band = (fq >= 70) & (fq <= 4000)
    win = np.hanning(fr)
    f0s, polys = [], []
    for s0 in starts:
        seg = x[s0:s0 + n]
        frames = [seg[i:i + fr] for i in range(0, len(seg) - fr + 1, hp)]
        e = np.array([float(np.sqrt(np.mean(f * f))) for f in frames]) if frames else np.zeros(0)
        if not len(frames) or e.max() < 1e-4:
            f0s.append(0); polys.append(0.0); continue
        F, P = [], []
        for f, en in zip(frames, e):
            if en < 0.35 * e.max():
                continue
            f0, ap = _yin(f, sr)
            pw = np.abs(np.fft.rfft(f * win, nfft)) ** 2
            tot = float(pw[band].sum()) or 1e-12
            hmask = np.zeros_like(band)
            hn = int(4000 // f0)
            for h in range(1, hn + 1):
                c = h * f0
                w = max(1.5 * df, 0.03 * c)
                hmask |= np.abs(fq - c) <= w
            hf = float(pw[band & hmask].sum()) / tot
            F.append(f0); P.append(0.5 * (1 - hf) + 0.5 * min(1.0, ap))
        f0s.append(int(round(float(np.median(F)))) if F else 0)
        polys.append(float(np.median(P)) if P else 0.0)
    return f0s, polys


def singers(path, work):
    """Divide la voce in finestre da 1,5 s e per ognuna calcola: l'impronta della voce (SpeechBrain ECAPA),
    l'altezza e la polifonia. Raggruppa in due cantanti e manda all'app anche i dati di ogni finestra,
    così l'app decide riga per riga e si corregge da sola dopo poche correzioni a mano."""
    import numpy as np
    import soundfile as sf
    import torch
    wav = os.path.join(work, "voci16k.wav")
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", "16000", wav])
    x, sr = sf.read(wav, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    win, hop = 1.5, 0.75
    n, h = int(win * sr), int(hop * sr)
    if len(x) < n:
        return {"v": 2, "duet": False, "hop": hop, "win": win, "k": "", "why": "voce troppo corta"}
    starts = list(range(0, len(x) - n + 1, h))
    rms = np.array([float(np.sqrt(np.mean(x[s0:s0 + n] ** 2))) for s0 in starts])
    thr = max(0.006, 0.18 * float(np.percentile(rms, 90)))
    act = np.where(rms > thr)[0]
    k = np.zeros(len(starts), dtype=int)
    if len(act) < 16:
        return {"v": 2, "duet": False, "hop": hop, "win": win, "k": "".join(map(str, k)), "why": "poca voce"}
    m = spk_model()
    emb = []
    with torch.no_grad():
        for b in range(0, len(act), 64):
            ids = act[b:b + 64]
            batch = torch.from_numpy(np.stack([x[starts[i]:starts[i] + n] for i in ids]))
            e = m.encode_batch(batch.to(m.device)).squeeze(1).float().cpu().numpy()
            emb.append(e)
    E = np.concatenate(emb, 0)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    f0s, polys = pitch_feats(x, sr, [starts[i] for i in act], n)
    # due gruppi (k-means sul coseno), partendo dalle due finestre più lontane
    mu = E.mean(0)
    a = int(np.argmin(E @ mu))
    b2 = int(np.argmin(E @ E[a]))
    C = np.stack([E[a], E[b2]])
    for _ in range(30):
        S = E @ C.T
        lab = S.argmax(1)
        newC = []
        for j in range(2):
            v = E[lab == j].mean(0) if np.any(lab == j) else C[j]
            newC.append(v / (np.linalg.norm(v) + 1e-9))
        newC = np.stack(newC)
        if np.allclose(newC, C, atol=1e-5):
            break
        C = newC
    # rifinitura: le voci medie si calcolano solo sui pezzi sicuri e poco polifonici,
    # così i pezzi cantati insieme e i cori non le sporcano
    P = np.array(polys)
    for _ in range(4):
        S = E @ C.T
        lab = S.argmax(1)
        gap = np.abs(S[:, 0] - S[:, 1])
        sure = (gap > np.percentile(gap, 35)) & (P <= np.percentile(P, 70))
        newC = []
        for j in range(2):
            sel = (lab == j) & sure
            v = E[sel].mean(0) if np.any(sel) else C[j]
            newC.append(v / (np.linalg.norm(v) + 1e-9))
        C = np.stack(newC)
    S = E @ C.T
    lab = S.argmax(1)
    sim = float(C[0] @ C[1])
    minor = float(min(np.mean(lab == 0), np.mean(lab == 1)))
    F0 = np.array(f0s, dtype=float)
    def med(j):
        v = F0[(lab == j) & (F0 > 0)]
        return float(np.median(v)) if len(v) else 0.0
    fa, fb = med(0), med(1)
    split = abs(12 * np.log2(fa / fb)) if fa > 0 and fb > 0 else 0.0  # semitoni fra le due voci
    duet = minor >= 0.1 and (sim < DUET_SIM or (split >= 4 and sim < 0.85))
    both = (np.abs(S[:, 0] - S[:, 1]) < max(BOTH_GAP, 0.22 * (1 - sim))) & (S.min(1) > 0.3)
    lab2 = np.where(both, 2, lab)
    first = next((int(v) for v in lab2 if v in (0, 1)), 0)
    ren = {first: 1, 1 - first: 2, 2: 3}
    for i, w in enumerate(act):
        k[w] = ren[int(lab2[i])] if duet else 1
    # dati di ogni finestra per l'app: impronta ridotta a 64 numeri da un byte, altezza, polifonia, volume
    D = 64
    Cm = E.mean(0)
    U, Sv, Vt = np.linalg.svd(E - Cm, full_matrices=False)
    D = min(D, Vt.shape[0])
    Z = (E - Cm) @ Vt[:D].T
    sc = float(np.abs(Z).max()) or 1.0
    q = np.clip(np.round(Z / sc * 127), -127, 127).astype(np.int8)
    return {"v": 2, "duet": bool(duet), "hop": hop, "win": win, "k": "".join(map(str, k)), "sim": round(sim, 3),
            "minor": round(minor, 3), "split": round(split, 1),
            "act": [int(i) for i in act], "d": D, "qs": round(sc / 127, 6),
            "emb": base64.b64encode(q.tobytes()).decode(),
            "f0": [int(v) for v in f0s], "poly": [int(round(v * 100)) for v in polys],
            "rms": [int(round(min(1.0, rms[i] / (rms.max() or 1)) * 1000)) for i in act]}


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


def _kbps(path):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=bit_rate", "-of", "default=nw=1:nk=1", path],
                           capture_output=True, text=True, timeout=60)
        return int(int(r.stdout.strip().split()[0]) / 1000)
    except Exception:
        return 96


def _shift_one(wav, dst, ratio, br, voice):
    enc = ["-c:a", "aac", "-b:a", "%dk" % br, dst]
    filters = ["rubberband=pitch=%.6f:pitchq=quality%s" % (ratio, ":formant=preserved" if voice else ""),
               "rubberband=pitch=%.6f" % ratio,
               # ultima riserva se ffmpeg non ha rubberband: cambia velocità e la riporta a tempo
               "asetrate=44100*%.6f,aresample=44100,atempo=%.6f" % (ratio, 1 / ratio)]
    last = None
    for f in filters:
        try:
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-af", f] + enc)
            return
        except RuntimeError as e:
            last = e
    raise last


def shift(job, inp):
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    work = tempfile.mkdtemp()
    try:
        files = inp.get("files") or []
        semis = [int(x) for x in (inp.get("semis") or []) if int(x) and -12 <= int(x) <= 12]
        if not files or not semis:
            return {"error": "mancano le tracce o i semitoni"}
        runpod.serverless.progress_update(job, "Preparo le tracce")
        src = []
        for f in files:
            k = re.sub(r"[^a-z0-9_]", "", os.path.splitext(os.path.basename(f.get("name") or "x"))[0].lower()) or "x"
            ext = os.path.splitext(f.get("name") or "")[1].lower() or ".m4a"
            p = os.path.join(work, k + ext)
            with open(p, "wb") as fh:
                fh.write(base64.b64decode(f.get("b64") or ""))
            wav = os.path.join(work, k + ".wav")
            run(["ffmpeg", "-y", "-loglevel", "error", "-i", p, "-ar", "44100", wav])
            src.append((k, wav, max(32, min(160, _kbps(p)))))
        tasks = [(k, wav, br, s) for s in semis for (k, wav, br) in src]
        runpod.serverless.progress_update(job, "Cambio tonalità a %d tracce" % len(tasks))

        def one(t):
            k, wav, br, s = t
            dst = os.path.join(work, "%s_%d.m4a" % (k, s))
            _shift_one(wav, dst, 2 ** (s / 12.0), br, k == "vocals")
            return t, dst
        out, total = {}, 0
        with ThreadPoolExecutor(max_workers=max(1, min(len(tasks), os.cpu_count() or 2))) as ex:
            for (k, wav, br, s), dst in ex.map(one, tasks):
                with open(dst, "rb") as fh:
                    b = base64.b64encode(fh.read()).decode()
                total += len(b)
                out.setdefault(str(s), {})[k] = b
        if total > LIMIT:
            return {"error": "TROPPO: risposta troppo grande, chiedi meno semitoni per volta"}
        return {"shift": out, "format": "m4a", "seconds": round(time.time() - t0, 1)}
    except Exception as e:
        print("ERRORE:", repr(e)[-3000:], flush=True)
        return {"error": str(e)[-700:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def handler(job):
    inp = job.get("input") or {}
    if inp.get("mode") == "shift":
        return shift(job, inp)
    if inp.get("mode") == "tone":
        # «Copia il suono del brano»: analisi della traccia di chitarra o basso (solo calcolo, niente modelli)
        from tone import tone_job
        return tone_job(job, inp, lambda m: runpod.serverless.progress_update(job, m))
    if inp.get("mode") == "video":
        # video finale: i file passano dal cloud R2 dell'utente (indirizzi firmati), non dentro la richiesta
        from video import video_job
        return video_job(job, inp, lambda m: runpod.serverless.progress_update(job, m))
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

        # --- chi canta: riconosce le due voci dei duetti ---
        if inp.get("mode") == "voices":
            runpod.serverless.progress_update(job, "Riconosco le voci")
            return {"singers": singers(src, work), "seconds": round(time.time() - t0, 1)}

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

        sing = None
        if inp.get("voices"):
            voc = os.path.join(out, model, "vocals.wav")
            if os.path.exists(voc):
                runpod.serverless.progress_update(job, "Riconosco le voci (duetti)")
                try:
                    sing = singers(voc, work)
                except Exception as e:  # la separazione resta valida
                    print("voci non riconosciute:", repr(e)[-500:], flush=True)
                    sing = {"error": str(e)[-300:]}

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
                "lyrics": lyrics, "singers": sing, "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # il messaggio torna all'app
        print("ERRORE:", repr(e)[-3000:], flush=True)
        return {"error": str(e)[-700:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)


runpod.serverless.start({"handler": handler})
