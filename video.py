"""
Pure Voltage - video finale su RunPod.

Rifà sul server, più velocemente del tempo reale, quello che il browser faceva fotogramma per fotogramma:
sfondo (MediaPipe selfie segmenter), trucco sul viso (MediaPipe face landmarker), duetto a schermo diviso,
testo karaoke, logo e titolo. In più corregge la sincronia labiale secondo per secondo.

I file grandi (video della fotocamera, audio finale, video dell'amico, risultato) passano dal cloud R2
dell'utente con indirizzi firmati: il worker non riceve mai le chiavi.
"""
import base64
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mpc

import numpy as np

MP_DIR = os.environ.get("PV_MP_MODELS", "/models/mp")
SEG_MODEL = os.path.join(MP_DIR, "selfie_segmenter.tflite")
FACE_MODEL = os.path.join(MP_DIR, "face_landmarker.task")
FONT_DIR = os.environ.get("PV_FONTS", "/models/fonts")
FAKE_ML = os.environ.get("PV_FAKE_ML") == "1"  # solo per le prove senza modelli


# ----------------------------------------------------------------------------- file
def download(url, dst, timeout=600):
    req = urllib.request.Request(url, headers={"User-Agent": "PureVoltage-worker"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 20)
    return dst


def upload(url, path, ctype="video/mp4", timeout=900):
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        req = urllib.request.Request(url, data=fh, method="PUT",
                                     headers={"Content-Type": ctype, "Content-Length": str(size)})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status >= 300:
                raise RuntimeError("upload HTTP %s" % r.status)


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "errore ffmpeg")[-1500:])
    return p.stdout


def duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", path])
    try:
        return float(out.strip().split()[0])
    except Exception:
        return 0.0


# ----------------------------------------------------------------------------- colori (come nel browser)
def css_filter(img):
    """contrast(1.06) saturate(1.1) brightness(1.02) del canvas, in un colpo solo."""
    c, s, b = 1.06, 1.1, 1.02
    m = np.array([[.213 + .787 * s, .715 - .715 * s, .072 - .072 * s],
                  [.213 - .213 * s, .715 + .285 * s, .072 - .072 * s],
                  [.213 - .213 * s, .715 - .715 * s, .072 + .928 * s]], np.float32) * (c * b)
    off = (0.5 - 0.5 * c) * b * m.sum(axis=1) / c
    out = img @ m.T + off
    return np.clip(out, 0, 1, out=out)


def soft_light(cb, cs):
    d = np.where(cb <= .25, ((16 * cb - 12) * cb + 4) * cb, np.sqrt(np.maximum(cb, 0)))
    return np.where(cs <= .5, cb - (1 - 2 * cs) * cb * (1 - cb), cb + (2 * cs - 1) * (d - cb))


def rgb2hsl(r, g, b):
    r, g, b = r / 255, g / 255, b / 255
    mx, mn = max(r, g, b), min(r, g, b)
    l = (mx + mn) / 2
    if mx == mn:
        return 0.0, 0.0, l
    d = mx - mn
    s = d / (2 - mx - mn) if l > .5 else d / (mx + mn)
    if mx == r:
        h = (g - b) / d + (6 if g < b else 0)
    elif mx == g:
        h = (b - r) / d + 2
    else:
        h = (r - g) / d + 4
    return h / 6, s, l


def hsl2rgb(h, s, l):
    if s == 0:
        v = l * 255
        return [v, v, v]

    def hue(p, q, t):
        t %= 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p
    q = l * (1 + s) if l < .5 else l + s - l * s
    p = 2 * l - q
    return [hue(p, q, h + 1 / 3) * 255, hue(p, q, h) * 255, hue(p, q, h - 1 / 3) * 255]


def bg_color(r, g, b):
    h, s, l = rgb2hsl(r, g, b)
    if s < .12:
        return hsl2rgb(.58, .18, .78) if l < .45 else hsl2rgb(.6, .25, .24)
    s = min(.55, max(.32, s * .8))
    l = .74 if l < .45 else .3
    return hsl2rgb(h, s, l)


def hexrgb(h, a=1.0):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)


# ----------------------------------------------------------------------------- caratteri
def font(kind, size):
    from PIL import ImageFont
    cand = {"lyr": [("JosefinSans.ttf", "Bold"), ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", None)],
            "title": [("Figtree.ttf", "SemiBold"), ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", None)],
            "dots": [("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", None)]}[kind]
    for path, var in cand:
        p = path if path.startswith("/") else os.path.join(FONT_DIR, path)
        if os.path.exists(p):
            f = ImageFont.truetype(p, max(6, int(size)))
            if var:
                try:
                    f.set_variation_by_name(var)
                except Exception:
                    pass
            return f
    return ImageFont.load_default()


# ----------------------------------------------------------------------------- testo karaoke (come recLyrics)
class Lyrics:
    def __init__(self, P, W, H):
        L = P.get("lyr") or {}
        self.on = bool(L.get("show")) and bool(L.get("lines"))
        self.lines = L.get("lines") or []
        self.off = float(L.get("off") or 0)
        self.mine = L.get("mine") or 0
        self.hi = hexrgb(L.get("hi") or "#6fd8ff")
        self.W, self.H = W, H
        self.base = round(min(W, H) * 0.062)
        self.yc = H * .5 if L.get("duoLay") else H * (0.79 if H > W else 0.8)
        self.cache = {}
        self.pc = {1: hexrgb("#3d8bff"), 2: hexrgb("#ff4d4d"), 3: hexrgb("#ffd23f")}

    def band(self):
        """fascia scura dietro al testo (statica)"""
        W, H, bh = self.W, self.H, self.base * 3.6
        y0, y1 = self.yc - bh * .75, self.yc - bh * .75 + bh * 1.8
        ys = np.arange(H, dtype=np.float32)
        u = (ys - y0) / max(1e-6, (bh * .75 + bh * 1.05))
        a = np.interp(u, [0, .22, .78, 1], [0, .55, .55, 0]).astype(np.float32)
        a[(ys < y0) | (ys >= y1)] = 0
        return a

    def _block(self, idx, size):
        key = (idx, size)
        if key in self.cache:
            return self.cache[key]
        from PIL import Image, ImageDraw, ImageFilter
        l = self.lines[idx]
        f = font("lyr", size)
        sp = f.getlength(" ")
        rows, cur, w = [], [], 0.0
        for wd in l.get("w") or []:
            ww = f.getlength(wd["s"])
            if cur and w + sp + ww > self.W * .88:
                rows.append(cur)
                cur, w = [], 0.0
            cur.append(dict(wd, ww=ww))
            w += (sp if len(cur) > 1 else 0) + ww
        if cur:
            rows.append(cur)
        lh = size * 1.22
        pad = int(size * 1.4)
        hgt = int(len(rows) * lh + 2 * pad)
        img = Image.new("L", (self.W, max(1, hgt)), 0)
        d = ImageDraw.Draw(img)
        boxes = []
        yy = pad + lh / 2  # centro della prima riga dentro al blocco
        for r in rows:
            rw = sum(x["ww"] for x in r) + sp * (len(r) - 1)
            x = (self.W - rw) / 2
            for wd in r:
                d.text((x, yy), wd["s"], font=f, fill=255, anchor="lm")
                boxes.append((x, wd["ww"], int(yy - size), int(yy + size), float(wd["t"]), float(wd["e"])))
                x += wd["ww"] + sp
            yy += lh
        m = np.asarray(img, np.float32) / 255
        sh = np.asarray(img.filter(ImageFilter.GaussianBlur(max(.5, size * .25 / 2))), np.float32) / 255 * .85
        blk = {"m": m, "sh": sh, "boxes": boxes, "h": len(rows) * lh, "pad": pad, "first": pad + lh / 2, "rows": len(rows)}
        self.cache[key] = blk
        return blk

    def _draw(self, out, idx, ycenter, size, alpha, prog, tt):
        l = self.lines[idx]
        p = l.get("p") or 0
        hil = self.pc.get(p, self.hi)
        if self.mine and p in (1, 2) and p != self.mine:
            alpha *= .45
        b = self._block(idx, size)
        lh = size * 1.22
        top = int(round(ycenter - (b["rows"] - 1) * lh / 2 - b["first"]))
        H = out.shape[0]
        y0, y1 = max(0, top), min(H, top + b["m"].shape[0])
        if y1 <= y0:
            return b["h"]
        m = b["m"][y0 - top:y1 - top]
        sh = b["sh"][y0 - top:y1 - top]
        reg = out[y0:y1]
        reg *= (1 - sh * alpha)[..., None]  # ombra nera
        col = self.pc.get(p) if (not prog and p in self.pc) else np.array([1, 1, 1], np.float32)
        a = (m * alpha)[..., None]
        reg[:] = reg * (1 - a) + col * a
        if prog:
            for (x, ww, ya, yb, t0, t1) in b["boxes"]:
                q = max(0.0, min(1.0, (tt - t0) / max(.08, t1 - t0)))
                if q <= 0:
                    continue
                xa, xb = int(x), int(math.ceil(x + ww * q))
                ra, rb = max(0, ya + top), min(H, yb + top)
                if rb <= ra or xb <= xa:
                    continue
                mm = b["m"][ra - top:rb - top, xa:xb] * alpha
                sub = out[ra:rb, xa:xb]
                sub[:] = sub * (1 - mm[..., None]) + hil * mm[..., None]
        return b["h"]

    def draw(self, out, t):
        if not self.on:
            return
        L, tt = self.lines, t - self.off
        i = -1
        for j, l in enumerate(L):
            if l["t"] <= tt:
                i = j
            else:
                break
        base, yc = self.base, self.yc
        if i >= 0 and tt <= L[i]["e"] + 1.2:
            h = self._draw(out, i, yc, base, 1.0, True, tt)
            if i + 1 < len(L):
                self._draw(out, i + 1, yc + h / 2 + base * 1.1, round(base * .78), .75, False, tt)
        elif i + 1 < len(L):
            n = L[i + 1]
            self._draw(out, i + 1, yc, round(base * .9), .85, False, tt)
            left = n["t"] - tt
            if 0 < left <= 3.2:
                k = max(1, math.ceil(left - .2))
                key = ("dots", k)
                if key not in self.cache:
                    from PIL import Image, ImageDraw
                    f = font("dots", base)
                    txt = " ".join(["●"] * k)
                    img = Image.new("L", (self.W, int(base * 2)), 0)
                    ImageDraw.Draw(img).text((self.W / 2, base), txt, font=f, fill=255, anchor="mm")
                    self.cache[key] = np.asarray(img, np.float32) / 255
                m = self.cache[key]
                top = int(yc - base * 1.6 - base)
                y0, y1 = max(0, top), min(out.shape[0], top + m.shape[0])
                if y1 > y0:
                    a = m[y0 - top:y1 - top][..., None]
                    out[y0:y1] = out[y0:y1] * (1 - a) + self.hi * a


# ----------------------------------------------------------------------------- parti fisse sopra al video
def static_layers(P, W, H):
    """A: sfumatura scura in basso (+ fascia del testo). B: logo e titolo. Ritorna (colore, alfa) premoltiplicati."""
    from PIL import Image, ImageDraw, ImageFilter
    ys = np.arange(H, dtype=np.float32)
    aA = np.where(ys >= H * .5, (ys - H * .5) / (H * .5) * .7, 0).astype(np.float32)
    A = np.repeat(aA[:, None], W, axis=1)
    lyr = Lyrics(P, W, H)
    if lyr.on:
        bA = lyr.band()[:, None]
        A = A + bA * (1 - A)  # due strati scuri uno sopra l'altro
    cA = np.zeros((H, W, 3), np.float32)  # nero
    # B: logo e titolo
    u = min(W, H)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if P.get("logo_b64") and not P.get("nologo"):
        try:
            lg = Image.open(io.BytesIO(base64.b64decode(P["logo_b64"]))).convert("RGBA")
            lw = u * .16
            lh = lw * lg.height / lg.width
            x0, y0 = u * .04, H - u * .04 - lh
            box = Image.new("RGBA", (int(lw + 12), int(lh + 12)), (255, 255, 255, int(255 * .9 * .9)))
            img.alpha_composite(box, (int(x0 - 6), int(y0 - 6)))
            lg = lg.resize((max(1, int(lw)), max(1, int(lh))), Image.LANCZOS)
            a = lg.split()[3].point(lambda v: int(v * .9))
            lg.putalpha(a)
            img.alpha_composite(lg, (int(x0), int(y0)))
        except Exception as e:
            print("logo:", e, flush=True)
    title = (P.get("title") or "").strip()
    if title:
        f = font("title", round(u * .032))
        txt = Image.new("L", (W, H), 0)
        ImageDraw.Draw(txt).text((W - u * .04, H - u * .04), title, font=f, fill=255, anchor="rd")
        sh = txt.filter(ImageFilter.GaussianBlur(3)).point(lambda v: int(v * .8 * .9))
        black = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        black.putalpha(sh)
        img.alpha_composite(black)
        white = Image.new("RGBA", (W, H), (255, 255, 255, 0))
        white.putalpha(txt.point(lambda v: int(v * .9)))
        img.alpha_composite(white)
    arr = np.asarray(img, np.float32) / 255
    return (cA, A), (arr[..., :3], arr[..., 3]), lyr


# ----------------------------------------------------------------------------- modelli
def load_models(need_seg, need_face):
    if FAKE_ML:
        return FakeSeg() if need_seg else None, FakeFace() if need_face else None
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    seg = face = None
    if need_seg:
        seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=mpt.BaseOptions(model_asset_path=SEG_MODEL), running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True, output_category_mask=False))
    if need_face:
        face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mpt.BaseOptions(model_asset_path=FACE_MODEL), running_mode=vision.RunningMode.VIDEO, num_faces=2))
    return seg, face


def mp_image(rgb8):
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb8))


def seg_mask(seg, rgb8, ts):
    if FAKE_ML:
        return seg.mask(rgb8)
    r = seg.segment_for_video(mp_image(rgb8), ts)
    m = r.confidence_masks[0].numpy_view()
    return np.array(m, np.float32).reshape(m.shape[0], m.shape[1])


def face_points(face, rgb8, ts):
    """lista di volti, ognuno array (478, 2) normalizzato 0..1"""
    if FAKE_ML:
        return face.points(rgb8)
    r = face.detect_for_video(mp_image(rgb8), ts)
    return [np.array([[p.x, p.y] for p in lm], np.float32) for lm in (r.face_landmarks or [])]


def face_index():
    if FAKE_ML:
        return FakeFace.IX
    from mediapipe.tasks.python import vision
    C = vision.FaceLandmarksConnections
    ix = lambda cs: sorted({i for c in cs for i in (c.start, c.end)})
    return {"oval": ix(C.FACE_LANDMARKS_FACE_OVAL), "le": ix(C.FACE_LANDMARKS_LEFT_EYE), "re": ix(C.FACE_LANDMARKS_RIGHT_EYE),
            "lb": ix(C.FACE_LANDMARKS_LEFT_EYEBROW), "rb": ix(C.FACE_LANDMARKS_RIGHT_EYEBROW), "lips": ix(C.FACE_LANDMARKS_LIPS)}


class FakeSeg:
    def mask(self, rgb):
        h, w = rgb.shape[:2]
        yy, xx = np.mgrid[0:256, 0:256] / 256.0
        return (((xx - .5) / .28) ** 2 + ((yy - .62) / .5) ** 2 < 1).astype(np.float32)


class FakeFace:
    IX = {"oval": list(range(0, 12)), "le": [12, 13, 14], "re": [15, 16, 17], "lb": [18, 19], "rb": [20, 21], "lips": [22, 23, 24, 25]}

    def points(self, rgb):
        pts = np.zeros((478, 2), np.float32) + [.5, .4]
        for i in range(12):
            a = i / 12 * 2 * math.pi
            pts[i] = [.5 + .12 * math.cos(a), .4 + .16 * math.sin(a)]
        pts[12:15] = [[.45, .36], [.47, .36], [.46, .37]]
        pts[15:18] = [[.53, .36], [.55, .36], [.54, .37]]
        pts[22:26] = [[.47, .48], [.53, .48], [.5, .47], [.5, .5]]
        pts[50] = [.43, .44]
        pts[280] = [.57, .44]
        pts[13] = [.5, .47]
        pts[14] = [.5, .47 + .01 * (1 + math.sin(time.time() * 9))]
        return [pts]


# ----------------------------------------------------------------------------- pittore (come makeFinalPainter)
class Painter:
    def __init__(self, P, W, H, seg, face):
        import cv2
        self.cv2 = cv2
        self.P, self.W, self.H, self.seg, self.face = P, W, H, seg, face
        self.ema = None
        self.inv = None
        self.kF = float(P.get("kF") or .6)
        self.beauty = float(P.get("beauty") or 0)
        self.bg = P.get("bg") or "real"
        self.col = [70.0, 90.0, 120.0]
        self.colT = -9.0
        self.lm = None
        self.fk = 0
        self.ix = face_index() if face else None
        self.bgimg = None
        if self.bg == "img" and P.get("bg_b64"):
            from PIL import Image
            im = Image.open(io.BytesIO(base64.b64decode(P["bg_b64"]))).convert("RGB").resize((W, H), Image.LANCZOS)
            self.bgimg = np.asarray(im, np.float32) / 255

    def _mask(self, frame8, ts):
        cv2 = self.cv2
        a = seg_mask(self.seg, frame8, ts)
        hh, w = a.shape
        if self.inv is None:
            ys, xs = np.mgrid[0:hh, 0:w]
            dx, dy = np.abs(xs / w - .5), ys / hh
            c = a[(dx < .15) & (dy > .35) & (dy < .8)]
            e = a[(dx > .45) & (dy < .3)]
            self.inv = bool(c.size and e.size and c.mean() < e.mean())
        if self.inv:
            a = 1 - a
        self.ema = a.copy() if self.ema is None or self.ema.shape != a.shape else self.ema * self.kF + a * (1 - self.kF)
        q = np.clip((self.ema - .35) / .3, 0, 1)
        q = q * q * (3 - 2 * q)
        m = cv2.resize(q, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(m, (0, 0), max(1, round(self.W / 400)))

    def _face(self, pts_list):
        """maschera del viso senza occhi, sopracciglia e bocca; più guance e labbra"""
        cv2, W, H = self.cv2, self.W, self.H
        m = np.zeros((H, W), np.uint8)
        info = []
        for pts in pts_list:
            px = pts * [W, H]

            def hull(k):
                p = px[self.ix[k]].astype(np.float32)
                return cv2.convexHull(p).reshape(-1, 2) if len(p) >= 3 else p

            def scaled(pp, sc):
                c = pp.mean(axis=0)
                return np.round(c + (pp - c) * sc).astype(np.int32)
            ov = hull("oval")
            cv2.fillPoly(m, [scaled(ov, 1.0)], 255)
            for k, sc in (("le", 1.5), ("re", 1.5), ("lb", 1.3), ("rb", 1.3), ("lips", 1.1)):
                cv2.fillPoly(m, [scaled(hull(k), sc)], 0)
            info.append({"lips": scaled(hull("lips"), 1.0), "w": float(ov[:, 0].max() - ov[:, 0].min()), "ch": [px[50], px[280]]})
        mf = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), max(2, round(W / 70)))
        return mf, info

    def paint(self, frame8, ts, t):
        """frame8: fotogramma già ritagliato e (se serve) specchiato, W×H RGB uint8. Ritorna float RGB 0..1"""
        cv2, W, H, beauty = self.cv2, self.W, self.H, self.beauty
        cam = css_filter(frame8.astype(np.float32) / 255)
        need = self.bg != "real" or beauty > 0
        mF = self._mask(frame8, ts) if (need and self.seg) else None
        FM = None
        if beauty > 0 and self.face:
            if self.fk % 2 == 0 or self.lm is None:
                try:
                    self.lm = face_points(self.face, frame8, ts)
                except Exception:
                    self.lm = []
            self.fk += 1
            if self.lm:
                FM = self._face(self.lm)
        if FM is not None:
            fm, info = FM
            sw, sh = max(16, round(W / 6)), max(16, round(H / 6))
            sm = cv2.resize(cv2.resize(cam, (sw, sh), interpolation=cv2.INTER_AREA), (W, H), interpolation=cv2.INTER_LINEAR)
            a = (fm * min(.85, beauty * 1.35))[..., None]
            cam = cam * (1 - a) + sm * a
            warm = hexrgb("#ffd2b8")
            a2 = (fm * beauty * .5)[..., None]
            cam = cam * (1 - a2) + soft_light(cam, warm) * a2
            if beauty >= .5:
                for f in info:
                    r = f["w"] * .16
                    for c in f["ch"]:
                        x0, x1 = int(max(0, c[0] - r)), int(min(W, c[0] + r))
                        y0, y1 = int(max(0, c[1] - r)), int(min(H, c[1] + r))
                        if x1 <= x0 or y1 <= y0:
                            continue
                        yy, xx = np.mgrid[y0:y1, x0:x1]
                        d = np.sqrt((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / max(1, r)
                        al = (np.clip(1 - d, 0, 1) * .55)[..., None].astype(np.float32)
                        sub = cam[y0:y1, x0:x1]
                        cam[y0:y1, x0:x1] = sub * (1 - al) + soft_light(sub, np.array([1, 110 / 255, 120 / 255], np.float32)) * al
                    lm_ = np.zeros((H, W), np.uint8)
                    cv2.fillPoly(lm_, [f["lips"]], 255)
                    la = (lm_.astype(np.float32) / 255 * .22)[..., None]
                    cam = cam * (1 - la) + cam * hexrgb("#d2506e") * la
        elif beauty > 0 and mF is not None:
            sw, sh = max(16, round(W / 6)), max(16, round(H / 6))
            small = cv2.resize(cam, (sw, sh), interpolation=cv2.INTER_AREA)
            msm = cv2.resize(mF, (sw, sh), interpolation=cv2.INTER_AREA)
            px = small * 255
            r, g, b = px[..., 0], px[..., 1], px[..., 2]
            cb = 128 - .168736 * r - .331264 * g + .5 * b
            cr = 128 + .5 * r - .418688 * g - .081312 * b
            sk2 = np.maximum(0, 1 - (((cb - 110) / 22) ** 2 + ((cr - 152) / 20) ** 2))
            sk = np.minimum(1, sk2 * 1.6 * msm)
            skb = cv2.resize(sk.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            smb = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
            a = (skb * min(.9, beauty * 1.3))[..., None]
            cam = cam * (1 - a) + smb * a
            a2 = (skb * beauty * .45)[..., None]
            cam = cam * (1 - a2) + soft_light(cam, hexrgb("#ffd2b8")) * a2
        if self.bg == "real" or mF is None:
            return cam
        if self.bgimg is not None:
            bgc = self.bgimg
        elif self.bg == "blur":
            bw, bh = max(8, round(W / 20)), max(8, round(H / 20))
            bgc = cv2.GaussianBlur(cv2.resize(cv2.resize(cam, (bw, bh), interpolation=cv2.INTER_AREA), (W, H)), (0, 0), 6)
        else:
            if t - self.colT > 1.2 or self.colT < 0:
                sw, sh = 64, max(2, round(64 * H / W))
                p = cv2.resize(cam, (sw, sh), interpolation=cv2.INTER_AREA) * 255
                mk = cv2.resize(mF, (sw, sh), interpolation=cv2.INTER_AREA) * 255
                rows = np.where((mk > 200).sum(axis=1) > sw * .05)[0]
                first = self.colT < 0
                self.colT = t
                if rows.size:
                    top, bot = rows[0], rows[-1]
                    sel = mk[round(top + (bot - top) * .5):bot + 1] > 220
                    pp = p[round(top + (bot - top) * .5):bot + 1][sel]
                    if len(pp) > 20:
                        tg = bg_color(*pp.mean(axis=0))
                        self.col = list(tg) if first else [c + (x - c) * .5 for c, x in zip(self.col, tg)]
            r, g, b = [round(c) for c in self.col]
            c0 = np.array([min(255, r + 28), min(255, g + 28), min(255, b + 28)], np.float32) / 255
            c1 = np.array([max(0, r - 22), max(0, g - 22), max(0, b - 22)], np.float32) / 255
            if getattr(self, "_grad", None) is None:
                yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
                r0, r1 = min(W, H) * .1, max(W, H) * .75
                d = np.sqrt((xx - W / 2) ** 2 + (yy - H * .45) ** 2)
                self._grad = np.clip((d - r0) / (r1 - r0), 0, 1)[..., None]
            bgc = c0 * (1 - self._grad) + c1 * self._grad
        a = mF[..., None]
        return bgc * (1 - a) + cam * a


# ----------------------------------------------------------------------------- lettura dei fotogrammi
class Frames:
    """legge un video in sequenza (fps fissi, ritagliato a W×H) e dà il fotogramma più vicino al tempo chiesto"""

    def __init__(self, path, t0, fps, W, H, mirror=False, cover=True):
        vf = ["fps=%s" % fps]
        if cover:
            vf += ["scale=%d:%d:force_original_aspect_ratio=increase" % (W, H), "crop=%d:%d" % (W, H)]
        else:
            vf += ["scale=%d:%d" % (W, H)]
        if mirror:
            vf.append("hflip")
        self.t0 = max(0.0, t0)
        self.fps, self.W, self.H = fps, W, H
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", "%.3f" % self.t0, "-i", path, "-vf", ",".join(vf),
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.n = -1
        self.buf = deque(maxlen=40)
        self.eof = False
        self.last = np.zeros((H, W, 3), np.uint8)

    def _next(self):
        if self.eof:
            return False
        sz = self.W * self.H * 3
        b = self.p.stdout.read(sz)
        if not b or len(b) < sz:
            self.eof = True
            return False
        self.n += 1
        self.last = np.frombuffer(b, np.uint8).reshape(self.H, self.W, 3)
        self.buf.append((self.n, self.last))
        return True

    def at(self, t):
        k = int(round((t - self.t0) * self.fps))
        if k < 0:
            k = 0
        while self.n < k and self._next():
            pass
        for n, fr in reversed(self.buf):
            if n <= k:
                return fr
        return self.buf[0][1] if self.buf else self.last

    def close(self):
        try:
            self.p.kill()
        except Exception:
            pass


# ----------------------------------------------------------------------------- sincronia labiale secondo per secondo
def mouth_track(path, P, t_from, t_to, fps_a=15):
    """apertura della bocca nel tempo del video grezzo (secondi, valore)"""
    W, H = 360, int(360 * P["H"] / P["W"]) // 2 * 2
    _, face = load_models(False, True)
    fr = Frames(path, t_from, fps_a, W, H, mirror=False)
    out = []
    t = t_from
    k = 0
    while t <= t_to:
        f = fr.at(t)
        if fr.eof and fr.n >= 0 and int(round((t - fr.t0) * fps_a)) > fr.n + 1:
            break
        try:
            pts = face_points(face, f, int(k * 1000 / fps_a) + 1)
        except Exception:
            pts = []
        if pts:
            p = pts[0]
            fh = max(1e-3, abs(p[152][1] - p[10][1]))
            out.append((t, float(abs(p[14][1] - p[13][1]) / fh)))
        t += 1.0 / fps_a
        k += 1
    fr.close()
    return out


def _mouth_job(a):
    path, P, t0, t1 = a
    try:
        return mouth_track(path, P, t0, t1)
    except Exception as e:
        print("bocca:", repr(e), flush=True)
        return []


def lipsync_curve(raw, P, dur, pool, nproc):
    """per ogni secondo, lo spostamento (s) che fa coincidere bocca e voce; poi lisciato"""
    env = np.array(P.get("env") or [], np.float32)
    hz = float(P.get("env_hz") or 50)
    if env.size < hz * 4:
        return None
    vstart, av = float(P.get("vstart") or 0), float(P.get("av") or 0)
    rt0, rt1 = vstart + max(0.0, -av) - .5, vstart + max(0.0, dur - av) + .5
    span = (rt1 - rt0) / nproc
    parts = [(raw, P, max(0.0, rt0 + i * span), rt0 + (i + 1) * span) for i in range(nproc)]
    mt = [x for part in pool.map(_mouth_job, parts) for x in part]
    if len(mt) < 30:
        return None
    mt.sort()
    mts = np.array([x[0] for x in mt], np.float32)
    mvs = np.array([x[1] for x in mt], np.float32)
    # inviluppo voce e bocca su una griglia comune a 50 Hz, nel tempo dell'audio
    g = 1.0 / hz
    tg = np.arange(0, dur, g, dtype=np.float32)
    e = np.interp(tg, np.arange(env.size) / hz, env)

    def mouth_at(ts):
        rt = vstart + np.maximum(0, ts - av)
        return np.interp(rt, mts, mvs, left=np.nan, right=np.nan)
    shifts = np.arange(-.3, .3001, .02, dtype=np.float32)
    secs, best = [], []
    for s in np.arange(1.5, dur - 1.5, 1.0):
        sel = (tg >= s - 1.5) & (tg < s + 1.5)
        ev = e[sel]
        if ev.std() < 1e-4:
            continue
        ev = (ev - ev.mean()) / ev.std()
        sc = []
        for d in shifts:
            mv = mouth_at(tg[sel] + d)
            ok = ~np.isnan(mv)
            if ok.sum() < len(mv) * .7 or np.nanstd(mv) < 1e-4:
                sc.append(-1)
                continue
            m2 = (mv[ok] - mv[ok].mean()) / mv[ok].std()
            sc.append(float(np.mean(ev[ok] * m2)))
        sc = np.array(sc)
        j = int(np.argmax(sc))
        if sc[j] > .25:
            secs.append(s)
            best.append(float(shifts[j]))
    if len(secs) < 3:
        return None
    best = np.array(best)
    # mediana mobile (5) e niente salti bruschi
    sm = np.array([np.median(best[max(0, i - 2):i + 3]) for i in range(len(best))])
    return {"t": [float(x) for x in secs], "d": [float(x) for x in sm]}


# ----------------------------------------------------------------------------- un pezzo del video, in parallelo
def render_chunk(a):
    (idx, f0, f1, P, raw, duo, work) = a
    W, H, fps = int(P["W"]), int(P["H"]), float(P["fps"])
    duo_on = bool(duo)
    HH = round(H / 2) if duo_on else H
    vstart, av = float(P.get("vstart") or 0), float(P.get("av") or 0)
    curve = P.get("lips")
    ct = np.array(curve["t"], np.float32) if curve else None
    cd = np.array(curve["d"], np.float32) if curve else None

    def rawt(t):
        x = vstart + max(0.0, t - av)
        if ct is not None:
            x += float(np.interp(t, ct, cd))
        return max(0.0, x)
    t_first = f0 / fps
    need_seg = (P.get("bg") or "real") != "real" or float(P.get("beauty") or 0) > 0
    seg, face = load_models(need_seg, float(P.get("beauty") or 0) > 0)
    painter = Painter(P, W, HH, seg, face)
    (cA, aA), (cB, aB), lyr = static_layers(P, W, H)
    fr = Frames(raw, rawt(t_first) - .6, fps, W, HH, mirror=bool(P.get("mirror")))
    fd = Frames(duo, float(P["duo_start"]) + t_first - .2, fps, W, H - HH, mirror=False) if duo_on else None
    out_path = os.path.join(work, "part_%03d.mp4" % idx)
    enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H),
                            "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                            "-pix_fmt", "yuv420p", "-r", str(fps), out_path], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    title_t0 = float(P.get("lyr_t0") or 0)
    for k in range(f0, f1):
        t = k / fps
        out = np.zeros((H, W, 3), np.float32)
        f = fr.at(rawt(t))
        body = painter.paint(f, int(k * 1000 / fps) + 1, t)
        out[H - HH:] = body
        if fd is not None:
            want = max(0.0, float(P["duo_start"]) + t)
            out[:H - HH] = fd.at(want).astype(np.float32) / 255
            out[H - HH - 1:H - HH + 1] = out[H - HH - 1:H - HH + 1] * .15 + .85
        out = out * (1 - aA[..., None]) + cA * aA[..., None]
        lyr.draw(out, title_t0 + t)
        out = out * (1 - aB[..., None]) + cB
        np.clip(out, 0, 1, out=out)
        enc.stdin.write((out * 255 + .5).astype(np.uint8).tobytes())
    enc.stdin.close()
    enc.wait()
    fr.close()
    if fd:
        fd.close()
    return out_path


# ----------------------------------------------------------------------------- lavoro completo
def video_job(job, inp, progress=lambda m: None):
    t0 = time.time()
    work = tempfile.mkdtemp()
    try:
        P = dict(inp.get("p") or {})
        progress("Scarico i file dal cloud")
        raw = download(inp["video_url"], os.path.join(work, "cam.bin"))
        aud = download(inp["audio_url"], os.path.join(work, "audio.bin"))
        duo = download(inp["duo_url"], os.path.join(work, "duo.bin")) if inp.get("duo_url") else None
        dur = duration(aud)
        if dur <= 0:
            return {"error": "audio finale non leggibile"}
        fps = float(P.get("fps") or 30)
        P["fps"] = fps
        res = os.path.join(work, "finale.mp4")
        if P.get("plain"):
            # niente da ridipingere: solo l'audio nuovo sul video registrato, con lo spostamento giusto
            vstart, av = float(P.get("vstart") or 0), float(P.get("av") or 0)
            start, pad = vstart + max(0.0, -av), max(0.0, av)
            vf = "trim=start=%.3f,setpts=PTS-STARTPTS,fps=%s" % (start, fps) + (",tpad=start_duration=%.3f:start_mode=clone" % pad if pad > 0 else "")
            progress("Unisco video e audio")
            run(["ffmpeg", "-y", "-v", "error", "-i", raw, "-i", aud, "-filter_complex", "[0:v]" + vf + "[v]", "-map", "[v]", "-map", "1:a",
                 "-t", "%.3f" % dur, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", res])
            lips = None
        else:
            nproc = max(1, min(int(os.environ.get("PV_PROCS") or os.cpu_count() or 2), 16))
            ctx = mpc.get_context("spawn")
            with ProcessPoolExecutor(max_workers=nproc, mp_context=ctx) as pool:
                lips = None
                if P.get("lipsync"):
                    progress("Controllo la sincronia labiale secondo per secondo")
                    try:
                        lips = lipsync_curve(raw, P, dur, pool, nproc)
                    except Exception as e:
                        print("sincronia labiale:", repr(e), flush=True)
                        lips = None
                P["lips"] = lips
                N = int(math.ceil(dur * fps))
                parts = max(1, min(nproc * 2, N // 60 or 1))
                step = int(math.ceil(N / parts))
                jobs = [(i, i * step, min(N, (i + 1) * step), P, raw, duo, work) for i in range(parts) if i * step < N]
                progress("Rifinisco il video (%d parti in parallelo)" % len(jobs))
                outs = list(pool.map(render_chunk, jobs))
            lst = os.path.join(work, "parti.txt")
            with open(lst, "w") as fh:
                fh.write("".join("file '%s'\n" % o for o in outs))
            progress("Unisco le parti e l'audio")
            run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-i", aud, "-map", "0:v", "-map", "1:a",
                 "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", "%.3f" % dur, "-movflags", "+faststart", res])
        progress("Carico il video finito nel cloud")
        upload(inp["out_url"], res)
        info = {"ok": True, "seconds": round(time.time() - t0, 1), "size": os.path.getsize(res)}
        if lips:
            info["lips"] = {"min_ms": round(min(lips["d"]) * 1000), "max_ms": round(max(lips["d"]) * 1000), "points": len(lips["d"])}
        return info
    except Exception as e:
        print("ERRORE video:", repr(e)[-3000:], flush=True)
        return {"error": str(e)[-700:]}
    finally:
        shutil.rmtree(work, ignore_errors=True)
