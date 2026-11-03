# Fonica Infinity - Demucs + Whisper su RunPod Serverless (scheda video NVIDIA)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# ffmpeg per l'audio, compilatore per i pacchetti che non arrivano già pronti
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel Cython
# torchaudio della stessa versione di torch, così non viene sostituito
RUN pip install "torchaudio==2.3.1" --index-url https://download.pytorch.org/whl/cu121
RUN pip install "demucs==4.0.1"
RUN pip install "openai-whisper==20240930"
RUN pip install "runpod>=1.6" soundfile

# scarica i modelli dentro l'immagine, così ogni avvio è più rapido
RUN python -c "from demucs.pretrained import get_model; [get_model(n) for n in ('htdemucs','htdemucs_6s','htdemucs_ft')]"
RUN python -c "import whisper; whisper.load_model('turbo', device='cpu')"

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
