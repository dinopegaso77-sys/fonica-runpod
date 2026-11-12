# Fonica Infinity - Demucs + Whisper su RunPod Serverless (scheda video NVIDIA)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
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
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
