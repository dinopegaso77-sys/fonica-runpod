"""
Pure Voltage - «Copia il suono del brano»: analisi della traccia separata di chitarra o basso.

Ingresso (mode "tone"):
  {"mode": "tone", "audio_url": "<indirizzo firmato R2 della traccia>", "role": "chitarra" | "basso", "bpm": 130}
Uscita:
  {"sounds": [{name, share, drive, cls, segs: [[da, a], ...], fx: {...pedali...}, curve: [...dB...]}],
   "freqs": [...Hz...], "fx": {...effetti comuni...}, "info": {...misure...}, "seconds": 12.3}

Misura senza intelligenza artificiale, solo analisi del segnale (numpy):
  - curva delle frequenze media (1/6 di ottava) → serve all'app per l'«EQ di somiglianza»
  - quanto è distorto (fattore di cresta, densità delle armoniche, dinamica)
  - parti del brano con suoni diversi (raggruppamento delle finestre di 3 secondi)
  - delay (autocorrelazione «sbiancata»: l'eco è una copia esatta, il musicista che ripete non lo è)
  - riverbero (coda dopo gli stop improvvisi)
  (chorus, phaser e flanger non ancora: si accendono a orecchio)
"""
import os, json, shutil, subprocess, tempfile, time, urllib.request
import numpy as np

SR = 22050
EPS = 1e-12
FREQS = [round(40 * 2 ** (k / 6), 1) for k in range(0, 50) if 40 * 2 ** (k / 6) <= 10500]


# ---------------------------------------------------------------- lettura
def fetch(url, dst, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "PureVoltage"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 20)


def decode(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()
    return x


# ---------------------------------------------------------------- strumenti
def frames(x, n, hop):
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    k = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(k)[:, None]
    return x[idx]


def rms_db(x, n=1024, hop=256):
    f = frames(x, n, hop)
    return 10 * np.log10(np.mean(f * f, axis=1) + EPS)


def band_db(power, fr):
    """potenza media per banda di 1/6 di ottava → dB"""
    out = []
    for c in FREQS:
        lo, hi = c * 2 ** (-1 / 12), c * 2 ** (1 / 12)
        m = (fr >= lo) & (fr < hi)
        if not m.any():
            m = np.argmin(np.abs(fr - c)) == np.arange(len(fr))
        out.append(10 * np.log10(np.mean(power[m]) + EPS))
    return np.array(out)


def norm_curve(c):
    f = np.array(FREQS)
    ref = c[(f >= 150) & (f <= 4000)].mean()
    return c - ref


# ---------------------------------------------------------------- distorsione
def drive_features(seg, fr, Pf):
    """seg: campioni della finestra; Pf: spettri di potenza dei singoli fotogrammi della finestra"""
    a = np.abs(seg)
    rms = np.sqrt(np.mean(seg * seg) + EPS)
    crest = 20 * np.log10(np.percentile(a, 99.9) / rms + EPS)
    m = (fr >= 1000) & (fr <= 4000)
    p = Pf[:, m] + EPS
    # quanta energia sta fuori dai picchi delle armoniche, fotogramma per fotogramma:
    # le note pulite hanno picchi netti, la distorsione (intermodulazione) riempie gli spazi
    lv = p.sum(1)
    p = p[lv >= np.percentile(lv, 50)]
    k = max(1, int(p.shape[1] * 0.04))
    top = np.sort(p, axis=1)[:, -k:].sum(1)
    flat = float(np.median(1 - top / p.sum(1)))
    P = Pf.mean(0)
    lo = P[(fr >= 100) & (fr < 1200)].sum() + EPS
    hi = P[(fr >= 1500) & (fr < 5000)].sum() + EPS
    bright = 10 * np.log10(hi / lo)
    e = rms_db(seg, 512, 256)
    e = e[e > e.max() - 40]
    dyn = float(np.percentile(e, 90) - np.percentile(e, 20)) if len(e) > 8 else 10.0
    return crest, flat, bright, dyn


def drive_score(crest, flat, dyn, role):
    # pesi tarati su chitarre sintetiche (pulito → crunch → distorto → high gain)
    s_c = np.clip((14.5 - crest) / 7.0, 0, 1)
    s_f = np.clip((flat - FL0) / FL1, 0, 1)
    s_d = np.clip((10 - dyn) / 8, 0, 1)
    s = 0.45 * s_c + 0.35 * s_f + 0.20 * s_d
    if role == "basso":
        s = s * 0.9
    return float(np.clip(s, 0, 1))


FL0, FL1 = 0.2, 0.5
def bass_feats(seg):
    """basso: la distorsione accende le frequenze alte per tutta la durata delle note;
    la batteria rimasta nella traccia le accende solo a colpi brevi (cassa, rullante, piatti).
    Per questo la misura usa la mediana fotogramma per fotogramma, che ignora i colpi,
    e la «puntualità» delle frequenze alte per riconoscere la batteria."""
    n, hop = 1024, 256
    F = frames(seg, n, hop) * np.hanning(n)
    P = np.abs(np.fft.rfft(F, axis=1)) ** 2
    fq = np.fft.rfftfreq(n, 1 / SR)
    lo = P[:, (fq >= 40) & (fq < 400)].sum(1) + EPS
    hi = P[:, (fq >= 1500) & (fq < 6000)].sum(1) + EPS
    keep = lo > np.percentile(lo, 30)
    r = 10 * np.log10(hi[keep] / lo[keep]) if keep.any() else np.array([-60.0])
    ratio = float(np.median(r))
    hd = 10 * np.log10(hi)
    spike = float(np.percentile(hd, 95) - np.median(hd))
    mean_r = float(10 * np.log10(hi[keep].sum() / lo[keep].sum())) if keep.any() else ratio
    return ratio, spike, mean_r


def bass_score(crest, ratio):
    s_r = np.clip((ratio + 34) / 22, 0, 1)
    s_c = np.clip((14.5 - crest) / 7.0, 0, 1)
    return float(np.clip(0.8 * s_r + 0.2 * s_c * s_r, 0, 1))


CUT = {"chitarra": (0.14, 0.45, 0.75), "basso": (0.2, 0.5, 0.8)}


def drive_class(s, role):
    a, b, c = CUT.get(role, CUT["chitarra"])
    return "Pulito" if s < a else "Crunch" if s < b else "Distorto" if s < c else "Molto distorto"


# ---------------------------------------------------------------- raggruppamento in suoni
def kmeans(X, k, it=40, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(it):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return lab, C


def group_sounds(F):
    """F: righe [drive, brillantezza dB, livello dB]. Un suono nuovo solo se cambia davvero:
    molto più o meno distorto, oppure più forte e più brillante (il classico assolo con il boost)."""
    n = len(F)
    best = np.zeros(n, int)
    if n < 8:
        return best
    X = np.column_stack([F[:, 0] / 0.1, F[:, 1] / 4.0, F[:, 2] / 6.0])
    for k in (2, 3):
        if n < 6 * k:
            break
        lab, _ = min((kmeans(X, k, seed=s) for s in range(4)),
                     key=lambda r: ((X - r[1][r[0]]) ** 2).sum())
        share = np.bincount(lab, minlength=k) / n
        if share.min() < (0.12 if k == 2 else 0.08):
            break
        C = np.array([np.median(F[lab == j], axis=0) for j in range(k)])
        ok = all(abs(C[i, 0] - C[j, 0]) > 0.28 or (abs(C[i, 2] - C[j, 2]) > 6 and abs(C[i, 1] - C[j, 1]) > 3)
                 for i in range(k) for j in range(i + 1, k))
        if not ok:
            break
        for i in range(1, n - 1):
            if lab[i - 1] == lab[i + 1] != lab[i]:
                lab[i] = lab[i - 1]
        best = lab
    return best


# ---------------------------------------------------------------- delay
def detect_delay(m, active_mask, bpm):
    """autocorrelazione sbiancata a pezzi di ~6 s: un'eco è una copia esatta e dà un picco stretto
    nello stesso punto in tutti i pezzi; il musicista che ripete una frase non è mai preciso al campione"""
    n = 1 << 17
    lo, hi = int(0.07 * SR), int(1.3 * SR)
    rows = []
    for s in range(0, len(m) - n, n // 2):
        if active_mask[s:s + n].mean() < 0.6:
            continue
        seg = m[s:s + n] * np.hanning(n)
        X = np.fft.rfft(seg, 2 * n)
        W = X / (np.abs(X) + 1e-9 * np.abs(X).max())
        r = np.fft.irfft(np.abs(W) ** 2 * (np.abs(X) > 1e-4 * np.abs(X).max()))[:hi + 40]
        r = r / (r[0] + EPS)
        rows.append(r)
    if len(rows) < 3:
        return None, {}
    R = np.median(np.array(rows), axis=0)
    # prominenza: il picco contro i dintorni (±15 ms), così la colorazione del suono non conta
    w = int(0.015 * SR)
    best, bi = 0.0, lo
    prom = np.zeros(len(R))
    for i in range(lo, hi):
        nb = np.r_[R[i - w:i - 3], R[i + 4:i + w]]
        prom[i] = R[i] - np.median(nb)
    i = int(np.argmax(prom[lo:hi])) + lo
    pk = float(prom[i])
    noise = float(np.median(np.abs(prom[lo:hi])) + 1e-6)
    z = pk / noise
    L = i / SR
    hits = float(np.mean([row[i - 2:i + 3].max() - np.median(row[i - w:i + w]) > 3 * noise for row in rows]))
    info = {"delay_peak": round(pk, 4), "delay_z": round(z, 1), "delay_s": round(L, 3), "delay_hits": round(hits, 2)}
    if z < 25 or pk < 0.01 or hits < 0.6:
        return None, info
    fb = 0.25
    if 2 * i < hi:
        p2 = float(prom[2 * i - 3:2 * i + 4].max())
        if p2 > 6 * noise:
            fb = float(np.clip(p2 / pk * 2.5, 0.1, 0.8))
    level = float(np.clip(pk * 25.0, 0.1, 0.8))
    note = ""
    if bpm:
        q = 60.0 / bpm
        for nm, mul in (("da 1/4", 1), ("da 1/8 col punto", 0.75), ("da 1/8", 0.5), ("terzina", 2 / 3), ("da 1/16", 0.25)):
            if abs(L - q * mul) < 0.03 * max(1, q * mul):
                note = nm
                break
    return {"time": L, "fb": fb, "level": level, "note": note}, info


# ---------------------------------------------------------------- riverbero
def detect_reverb(m):
    """dopo uno stop secco: la mano ferma le corde in pochi centesimi di secondo (caduta velocissima),
    poi resta solo la stanza, che scende più piano. Misuro quanto è alta e quanto dura quella coda."""
    hop = 128
    fps = SR / hop
    e = rms_db(m, 512, hop)
    sm = np.convolve(e, np.ones(3) / 3, mode="same")
    floor = np.percentile(sm, 3)
    top = np.percentile(sm, 99)
    fast = 250.0                                  # dB al secondo: più veloce di qualsiasi stanza
    sl = np.r_[(sm[:-3] - sm[3:]) * fps / 3, np.zeros(3)]
    tails, rts = [], []
    i = 20
    while i < len(sm) - int(1.0 * fps):
        if sl[i] > fast and sm[i] > top - 22 and sm[i] >= sm[i - 20:i].max() - 4:
            pre = sm[i - 20:i + 1].max()
            j = i
            while j < i + int(0.2 * fps) and sl[j] > fast:
                j += 1
            drop = pre - sm[j]
            if drop < 5:
                i = j + 1
                continue
            j += 2
            seg = sm[j:j + int(0.7 * fps)]
            stop = np.where(seg < floor + 5)[0]
            if len(stop):
                seg = seg[:stop[0]]
            tail = float(sm[j] - pre)
            if sm[j] < floor + 5:
                tail = min(tail, -45.0)
            slope = 0.0
            if len(seg) >= 8:
                t = np.arange(len(seg)) / fps
                slope = np.polyfit(t, seg, 1)[0]
            if tail > -45 and slope > -12:
                # dopo la caduta il suono resta lì: era solo l'attacco di una nota, non uno stop
                i = j + 1
                continue
            if slope < -3:
                rts.append(-60.0 / slope)
            tails.append(tail)
            i = j + int(0.6 * fps)
        else:
            i += 1
    info = {"stops": len(tails)}
    if len(tails) < 2:
        return None, info
    tail = float(np.median(tails))
    rt = float(np.median(rts)) if len(rts) >= 2 else 0.5
    rt = float(np.clip(rt, 0.3, 5.0))
    info.update(tail_db=round(tail, 1), rt60=round(rt, 2))
    if tail < -34:
        return {"time": rt, "level": 0.0, "tail": tail}, info
    level = float(np.clip((tail + 36) / 26 * 0.6, 0.06, 0.65))
    return {"time": rt, "level": level, "tail": tail}, info


# ---------------------------------------------------------------- dai numeri ai pedali dell'app
def knob_rate(hz, mul):
    # nell'app: frequenza = 0.08 * 75^v * mul
    return float(np.clip(np.log(max(hz, 0.05) / (0.08 * mul)) / np.log(75), 0, 1))


def pedals(role, s, dyn, dly, rev, mod):
    fx = {}
    if role == "basso":
        fx["comp"] = {"on": 1, "sus": float(np.clip((16 - dyn) / 16, 0.35, 0.75)), "att": 0.25, "level": 0.5}
        if s >= 0.2:
            fx["od"] = {"on": 1, "drive": float(np.clip((s - 0.2) * 1.4, 0.15, 0.8)), "tone": 0.45, "level": 0.55}
    else:
        if s < 0.14:
            if dyn < 7:
                fx["comp"] = {"on": 1, "sus": float(np.clip((12 - dyn) / 10, 0.3, 0.7)), "att": 0.3, "level": 0.5}
        elif s < 0.45:
            fx["od"] = {"on": 1, "drive": float(0.15 + (s - 0.14) / 0.31 * 0.45), "tone": 0.5, "level": 0.5}
        elif s < 0.75:
            fx["ds"] = {"on": 1, "dist": float(0.3 + (s - 0.45) / 0.30 * 0.4), "tone": 0.5, "level": 0.45}
        else:
            fx["ds"] = {"on": 1, "dist": float(np.clip(0.72 + (s - 0.75), 0.72, 0.95)), "tone": 0.5, "level": 0.42}
            fx["gate"] = {"on": 1, "thr": 0.22, "rel": 0.35}
    if dly:
        fx["delay"] = {"on": 1, "time": float(np.clip((dly["time"] - 0.04) / 1.16, 0, 1)),
                       "fb": float(np.clip(dly["fb"] / 0.85, 0.05, 0.9)), "level": dly["level"]}
    if rev and rev["level"] > 0:
        fx["rev"] = {"on": 1, "time": float(np.clip((rev["time"] - 0.6) / 4.2, 0, 1)), "tone": 0.55, "level": rev["level"]}
    if mod and role != "basso":
        fx["chorus"] = {"on": 1, "rate": knob_rate(mod["rate"], 0.5), "depth": float(np.clip(mod["cents"] / 22, 0.25, 0.9)), "mix": 0.5}
    for v in fx.values():
        for k in v:
            v[k] = round(float(v[k]), 3)
    return fx


def tstr(t):
    t = int(round(t))
    return f"{t // 60}:{t % 60:02d}"


# ---------------------------------------------------------------- lavoro completo
def analyze(x, role="chitarra", bpm=None, say=lambda m: None):
    m = x.mean(0)
    pk = np.abs(m).max()
    if pk < 1e-4:
        raise ValueError("la traccia è vuota")
    m = m / pk * 0.9
    say("Cerco dove suona")
    e = rms_db(m, 1024, 256)
    thr = max(np.percentile(e, 99) - 30, -60)
    act = e > thr
    act_s = np.repeat(act, 256)[:len(m)]
    act_s = np.pad(act_s, (0, len(m) - len(act_s)))
    if act.mean() < 0.03:
        raise ValueError("in questa traccia lo strumento quasi non c'è")

    say("Misuro le frequenze")
    n, hop = 4096, 1024
    fr = np.fft.rfftfreq(n, 1 / SR)
    Fm = frames(m, n, hop)
    P = np.abs(np.fft.rfft(Fm * np.hanning(n), axis=1)) ** 2
    fa = e[np.minimum((np.arange(len(P)) * hop + n // 2) // 256, len(e) - 1)] > thr
    total = norm_curve(band_db(P[fa].mean(0), fr))

    say("Riconosco quanto è distorto e le parti del brano")
    W, H = int(3 * SR), int(1.5 * SR)
    wins = []
    for s in range(0, max(1, len(m) - W), H):
        if act_s[s:s + W].mean() < 0.6:
            continue
        seg = m[s:s + W]
        sel = act_s[s:s + W].astype(bool)
        a, b = s // hop, min(len(P), (s + W) // hop)
        pw = P[a:b][fa[a:b]] if fa[a:b].any() else P[a:b]
        Pw = pw.mean(0)
        crest, flat, bright, dyn = drive_features(seg[sel], fr, pw)
        lvl = 10 * np.log10(np.mean(seg[sel] ** 2) + EPS)
        w = dict(t=s / SR, crest=crest, flat=flat, bright=bright, dyn=dyn, lvl=lvl, P=Pw)
        if role == "basso":
            ratio, spike, mean_r = bass_feats(seg)
            w.update(d=bass_score(crest, ratio), ratio=ratio, spike=spike, bleed=bool(spike > 14 and mean_r - ratio > 5))
        else:
            w["d"] = drive_score(crest, flat, dyn, role)
        wins.append(w)
    if not wins:
        raise ValueError("lo strumento suona a pezzi troppo corti per capirne il suono")
    Fw = np.array([[w["d"], w["bright"], w["lvl"]] for w in wins])
    lab = group_sounds(Fw)

    say("Cerco delay e riverbero")
    # sul basso le note basse si ripetono identiche a ogni periodo e ingannano la ricerca dell'eco: il delay sul basso è rarissimo
    dly, i1 = (None, {}) if role == "basso" else detect_delay(m, act_s, bpm)
    rev, i2 = detect_reverb(m)
    mod, i3 = None, {}

    sounds = []
    for k in range(lab.max() + 1):
        ws = [w for w, l in zip(wins, lab) if l == k]
        s = float(np.median([w["d"] for w in ws]))
        dyn = float(np.median([w["dyn"] for w in ws]))
        segs = []
        for w in ws:
            if segs and w["t"] - segs[-1][1] <= 3.1:
                segs[-1][1] = w["t"] + 3
            else:
                segs.append([w["t"], w["t"] + 3])
        sounds.append(dict(
            share=round(len(ws) / len(wins), 2), drive=round(s, 3), cls=drive_class(s, role),
            lvl=float(np.median([w["lvl"] for w in ws])), bright=round(float(np.median([w["bright"] for w in ws])), 1),
            crest=round(float(np.median([w["crest"] for w in ws])), 1), flat=round(float(np.median([w["flat"] for w in ws])), 3), dyn=round(dyn, 1),
            segs=[[round(a, 1), round(b, 1)] for a, b in segs],
            fx=pedals(role, s, dyn, dly, rev, mod),
            curve=[round(float(v), 1) for v in norm_curve(band_db(np.mean([w["P"] for w in ws], axis=0), fr))]))
    # il suono principale per primo, poi in ordine di comparsa
    sounds.sort(key=lambda d: -d["share"])
    main = sounds[0]
    rest = sorted(sounds[1:], key=lambda d: d["segs"][0][0])
    sounds = [main] + rest
    top = max(d["lvl"] for d in sounds)
    used = {}
    for d in sounds:
        nm = d["cls"]
        if nm in used:
            nm += " (" + ("più forte" if d["lvl"] > used[nm] + 2 else "più brillante" if d["bright"] > 0 else "da " + tstr(d["segs"][0][0])) + ")"
        used.setdefault(d["cls"], d["lvl"])
        d["name"] = nm
        d["lvl"] = round(d["lvl"] - top, 1)
    fxc = {}
    if dly:
        fxc["delay"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in dly.items()}
    if rev:
        fxc["rev"] = {k: round(v, 2) for k, v in rev.items()}
    if mod:
        fxc["chorus"] = {k: round(v, 2) for k, v in mod.items()}
    info = {"active": round(float(act.mean()), 2), **i1, **i2, **i3}
    warn = ""
    if role == "basso":
        bl = float(np.mean([w.get("bleed", False) for w in wins]))
        info.update(bleed=round(bl, 2), spike=round(float(np.median([w["spike"] for w in wins])), 1), ratio=round(float(np.median([w["ratio"] for w in wins])), 1))
        if bl > 0.35:
            warn = "bleed"
    if warn:
        info["warn"] = warn
    return {"warn": warn, "sounds": sounds, "freqs": FREQS, "curve": [round(float(v), 1) for v in total], "effects": fxc, "info": info,
            "dur": round(len(m) / SR, 1)}


def tone_job(job, inp, progress):
    t0 = time.time()
    url = inp.get("audio_url")
    if not url:
        return {"error": "manca la traccia da analizzare"}
    role = inp.get("role") if inp.get("role") in ("chitarra", "basso") else "chitarra"
    bpm = inp.get("bpm") or None
    work = tempfile.mkdtemp()
    try:
        progress("Scarico la traccia")
        src = os.path.join(work, "traccia")
        fetch(url, src)
        x = decode(src)
        out = analyze(x, role, float(bpm) if bpm else None, progress)
        out["seconds"] = round(time.time() - t0, 1)
        return out
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        print("ERRORE tone:", repr(e)[-2000:], flush=True)
        return {"error": str(e)[-500:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)
