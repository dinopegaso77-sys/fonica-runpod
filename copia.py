"""
Pure Voltage - copia i programmi nella cartella Hugging Face di Pure Voltage (si fa una volta sola).

Lanciato da handler.py con la modalità "hfsync", nell'ambiente /opt/qwen:
    /opt/qwen/bin/python /copia.py richiesta.json risposta.json
richiesta: {"repo": "Purevoltage/pure-voltage-models", "force": false}
La chiave Hugging Face (permesso Write) arriva nella variabile HF_TOKEN: non viene mai scritta né restituita.
Ogni programma già presente nella cartella viene saltato, quindi si può rilanciare senza problemi.
"""
import json
import os
import shutil
import sys
import tempfile

# cartella nella copia  ->  originale su Hugging Face
SOURCES = {
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-forcedaligner-0.6b": "Qwen/Qwen3-ForcedAligner-0.6B",
    "bs-roformer-sw": "enerjazzer/BS-ROFO-SW-Fixed",
}

README = """---
license: other
tags:
- audio
- automatic-speech-recognition
---
# Pure Voltage - modelli

Copia dei modelli usati dal worker RunPod di Pure Voltage, senza modifiche.

| Cartella | Originale | Licenza |
|---|---|---|
| `qwen3-asr-1.7b` | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | Apache 2.0 |
| `qwen3-forcedaligner-0.6b` | [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | Apache 2.0 |
| `bs-roformer-sw` | BS-RoFormer SW di jarredou ([copia di enerjazzer](https://huggingface.co/enerjazzer/BS-ROFO-SW-Fixed)) | CC BY-NC-SA 4.0 (non commerciale) |

Tutti i diritti restano agli autori originali (Qwen Team, Alibaba Cloud; jarredou).
"""


def say(msg):
    print("@@ " + msg, flush=True)


def main(req_path, out_path):
    req = json.load(open(req_path))
    repo = (req.get("repo") or "Purevoltage/pure-voltage-models").strip()
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token.startswith("hf_"):
        raise RuntimeError("manca la chiave Hugging Face (inizia con hf_)")
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=token)
    who = api.whoami().get("name")
    say("Chiave valida (utente %s)" % who)
    have = set(api.list_repo_files(repo))
    done, skipped = [], []
    for sub, src in SOURCES.items():
        if not req.get("force") and any(f.startswith(sub + "/") for f in have):
            skipped.append(sub)
            continue
        tmp = tempfile.mkdtemp(dir="/tmp")
        try:
            say("Scarico %s" % src)
            snapshot_download(src, local_dir=tmp)
            shutil.rmtree(os.path.join(tmp, ".cache"), ignore_errors=True)
            say("Carico %s nella cartella %s" % (sub, repo))
            api.upload_folder(repo_id=repo, folder_path=tmp, path_in_repo=sub,
                              commit_message="Copia di %s" % src, ignore_patterns=[".cache/*", ".cache/**"])
            done.append(sub)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    api.upload_file(path_or_fileobj=README.encode(), path_in_repo="README.md", repo_id=repo,
                    commit_message="Descrizione e crediti")
    json.dump({"ok": True, "repo": repo, "copied": done, "already": skipped}, open(out_path, "w"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
