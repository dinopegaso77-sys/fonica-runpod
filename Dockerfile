# Pure Voltage - Demucs + Whisper su RunPod Serverless (scheda video NVIDIA)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# versioni bloccate per tutte le installazioni: una ricostruzione futura non può prendere librerie che non vanno d'accordo
COPY vincoli.txt /vincoli.txt
ENV PIP_CONSTRAINT=/vincoli.txt
# ffmpeg per l'audio, compilatore per i pacchetti che non arrivano già pronti
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel Cython "setuptools<70"
# torchaudio della stessa versione di torch, così non viene sostituito
RUN pip install "torchaudio==2.3.1" --index-url https://download.pytorch.org/whl/cu121
RUN pip install "demucs==4.0.1"
# whisper si costruisce con il setuptools già installato (le versioni nuove non lo compilano)
RUN pip install --no-build-isolation "openai-whisper==20240930"
RUN pip install "runpod>=1.6" soundfile
# riconoscimento delle voci per i duetti (SpeechBrain ECAPA)
RUN pip install "speechbrain==1.0.3" "huggingface_hub>=0.20,<0.27"

# testi delle canzoni con Qwen3-ASR: ambiente separato /opt/qwen che usa torch del sistema ma ha le sue librerie
# (transformers e huggingface_hub più nuovi), così Demucs, Whisper e SpeechBrain non cambiano.
# I modelli NON sono nell'immagine: arrivano dalla cartella Hugging Face di Pure Voltage (cache di RunPod o download).
COPY vincoli-qwen.txt /vincoli-qwen.txt
RUN apt-get update && apt-get install -y --no-install-recommends sox && rm -rf /var/lib/apt/lists/*
RUN python -m venv --system-site-packages /opt/qwen \
    && PIP_CONSTRAINT=/vincoli-qwen.txt /opt/qwen/bin/pip install --no-cache-dir "qwen-asr==0.0.6" hf_xet
# separatore nuovo BS-RoFormer SW (6 tracce) nello stesso ambiente. Se non si installa, la costruzione continua:
# il worker userà Demucs come sempre (lo dice il controllo finale qui sotto)
RUN PIP_CONSTRAINT=/vincoli-qwen.txt /opt/qwen/bin/pip install --no-cache-dir "bs-roformer-infer==0.1.5" \
    || echo "ATTENZIONE: separatore nuovo non installato, resta Demucs"

# ritmo e batteria più precisi: Beat This! (battiti e battute, MIT) e ADTOF (batteria, CC BY-NC-SA).
# Stanno in un ambiente TUTTO LORO (/opt/ritmo): così non possono mai cambiare le librerie di Qwen3 (testi) né del sistema.
# Se non si installano, la costruzione continua: l'app usa il metodo che ha già sul dispositivo.
RUN python -m venv --system-site-packages /opt/ritmo \
    && PIP_CONSTRAINT=/vincoli.txt /opt/ritmo/bin/pip install --no-cache-dir -c /vincoli-qwen.txt tqdm einops soxr rotary-embedding-torch \
       "https://github.com/CPJKU/beat_this/archive/main.zip" \
    && /opt/ritmo/bin/python -c "from beat_this.inference import File2Beats; File2Beats(checkpoint_path='final0', device='cpu', dbn=False); print('beat this ok')" \
    || echo "ATTENZIONE: Beat This! non installato, l'app calcola i battiti da sola"
RUN PIP_CONSTRAINT=/vincoli.txt /opt/ritmo/bin/pip install --no-cache-dir -c /vincoli-qwen.txt pretty_midi "git+https://github.com/xavriley/ADTOF-pytorch" \
    && /opt/ritmo/bin/python -c "import adtof_pytorch; print('adtof ok')" \
    || echo "ATTENZIONE: ADTOF non installato, l'app scrive la batteria da sola"

# scarica i modelli dentro l'immagine, così ogni avvio è più rapido
RUN python -c "from demucs.pretrained import get_model; [get_model(n) for n in ('htdemucs','htdemucs_6s','htdemucs_ft')]"
RUN python -c "import whisper; whisper.load_model('turbo', device='cpu')"
RUN python -c "from speechbrain.inference.speaker import EncoderClassifier; EncoderClassifier.from_hparams(source='speechbrain/spkrec-ecapa-voxceleb', savedir='/models/ecapa')"

# video finale: sfondo e trucco con MediaPipe (stessi modelli dell'app), disegno con OpenCV e Pillow
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
RUN pip install "numpy<2" "mediapipe==0.10.14" "jax<0.4.31" "jaxlib<0.4.31" "opencv-contrib-python<4.11" "opencv-python-headless<4.11" pillow
RUN mkdir -p /models/mp /models/fonts \
    && curl -fsSL -o /models/mp/selfie_segmenter.tflite https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite \
    && curl -fsSL -o /models/mp/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
    && curl -fsSL -o /models/fonts/JosefinSans.ttf "https://github.com/google/fonts/raw/main/ofl/josefinsans/JosefinSans%5Bwght%5D.ttf" \
    && curl -fsSL -o /models/fonts/Figtree.ttf "https://github.com/google/fonts/raw/main/ofl/figtree/Figtree%5Bwght%5D.ttf"
RUN python -c "import mediapipe, cv2, PIL; from mediapipe.tasks.python import vision; print('video ok', mediapipe.__version__, cv2.__version__)"

COPY video.py /video.py
COPY tone.py /tone.py
COPY testi.py /testi.py
COPY copia.py /copia.py
COPY separa.py /separa.py
COPY ritmo.py /ritmo.py
COPY handler.py /handler.py

# controllo finale: se una libreria non si carica, la costruzione si ferma qui e RunPod non riceve un worker guasto
RUN python -c "import numpy, scipy, torch, torchaudio, demucs, whisper, soundfile, runpod, speechbrain, cv2, mediapipe; print('controllo ok', 'numpy', numpy.__version__, 'torch', torch.__version__, 'runpod', runpod.__version__)" \
    && python -m demucs --help > /dev/null \
    && /opt/qwen/bin/python -c "import torch, transformers, huggingface_hub, qwen_asr; from qwen_asr import Qwen3ASRModel; print('qwen ok', 'torch', torch.__version__, 'transformers', transformers.__version__, 'hub', huggingface_hub.__version__)" \
    && python -c "import huggingface_hub, speechbrain; print('sistema ok', 'hub', huggingface_hub.__version__)" \
    && (/opt/ritmo/bin/python -c "import beat_this, adtof_pytorch; print('ritmo nuovo ok')" || echo "ritmo nuovo assente: l'app usa il suo metodo") \
    && (/opt/qwen/bin/python -c "import bs_roformer; print('separatore nuovo ok')" && /opt/qwen/bin/bs-roformer-infer --help | head -40 || echo "separatore nuovo assente: si usa Demucs") \
    && cd / && python -c "import ast; [ast.parse(open(f).read()) for f in ('/handler.py', '/testi.py', '/copia.py', '/separa.py', '/ritmo.py')]; import tone, video; print('worker ok')"
CMD ["python", "-u", "/handler.py"]
