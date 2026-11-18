"""
Pure Voltage - separazione in 6 tracce con BS-RoFormer SW (al posto di Demucs, che resta come riserva).

Gira nell'ambiente separato /opt/qwen, lanciato da handler.py:
    /opt/qwen/bin/python /separa.py richiesta.json risposta.json
richiesta: {"audio": "brano.mp3", "out": "cartella dove mettere vocals.wav, bass.wav, drums.wav, guitar.wav, piano.wav, other.wav"}

Modello: BS-RoFormer SW (6 tracce) di jarredou, licenza CC BY-NC-SA 4.0 (uso non commerciale).
Si cerca prima nella cartella Hugging Face di Pure Voltage (cache di RunPod o disco), altrimenti si scarica
dall'originale (circa 700 MB, una volta per worker acceso).
Qualsiasi errore fa uscire con codice diverso da 0: handler.py usa Demucs come sempre.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.environ.get("PV_MODELS", "Purevoltage/pure-voltage-models")
SUB = "bs-roformer-sw"
SRC = "enerjazzer/BS-ROFO-SW-Fixed"
CKPT, CONF = "BS-Rofo-SW-Fixed.ckpt", "BS-Rofo-SW-Fixed.yaml"
LOCAL = "/tmp/pvmodels"
STEMS = ("vocals", "bass", "drums", "guitar", "piano", "other")


def say(msg):
    print("@@ " + msg, flush=True)


def assets():
    root = "/runpod-volume/huggingface-cache/hub/models--" + REPO.replace("/", "--") + "/snapshots"
    for d in sorted(glob.glob(root + "/*"), key=os.path.getmtime, reverse=True) + [LOCAL]:
        ck, cf = os.path.join(d, SUB, CKPT), os.path.join(d, SUB, CONF)
        if os.path.exists(ck) and os.path.exists(cf):
            return ck, cf
    say("Prima volta su questo worker: scarico il separatore nuovo (circa un minuto)")
    from huggingface_hub import hf_hub_download
    dst = os.path.join(LOCAL, SUB)
    ck = hf_hub_download(SRC, CKPT, local_dir=dst)
    cf = hf_hub_download(SRC, CONF, local_dir=dst)
    return ck, cf


def main(req_path, out_path):
    t0 = time.time()
    req = json.load(open(req_path))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("serve la scheda video")
    ck, cf = assets()
    work = req["out"] + "_lavoro"
    inp, sto = os.path.join(work, "in"), os.path.join(work, "out")
    os.makedirs(inp, exist_ok=True)
    os.makedirs(sto, exist_ok=True)
    os.makedirs(req["out"], exist_ok=True)
    # il programma lavora su una cartella di WAV a 44,1 kHz stereo
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", req["audio"], "-ac", "2", "-ar", "44100",
                    os.path.join(inp, "brano.wav")], check=True, timeout=300)
    say("Separo le tracce con il separatore nuovo")
    exe = os.path.join(os.path.dirname(sys.executable), "bs-roformer-infer")
    cmd = [exe, "--config_path", cf, "--model_path", ck, "--input_folder", inp, "--store_dir", sto]
    pr = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
    print(pr.stdout[-3000:], pr.stderr[-3000:], flush=True)
    if pr.returncode != 0:
        raise RuntimeError("bs-roformer-infer: " + (pr.stderr or pr.stdout)[-800:])
    found = {}
    for w in glob.glob(os.path.join(sto, "**", "*.wav"), recursive=True):
        n = os.path.splitext(os.path.basename(w))[0].lower()
        if "instrument" in n:
            continue
        for s in STEMS:
            if n == s or n.endswith("_" + s) or n.endswith("-" + s) or n.endswith(" " + s):
                found[s] = w
    missing = [s for s in ("vocals", "bass", "drums", "other") if s not in found]
    if missing:
        raise RuntimeError("tracce mancanti: %s (file: %s)" % (missing, [os.path.basename(w) for w in
                           glob.glob(os.path.join(sto, "**", "*.wav"), recursive=True)][:12]))
    for s, w in found.items():
        shutil.move(w, os.path.join(req["out"], s + ".wav"))
    shutil.rmtree(work, ignore_errors=True)
    json.dump({"ok": True, "stems": sorted(found), "model": "bs_roformer_sw", "seconds": round(time.time() - t0, 1)},
              open(out_path, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
