# Fonica Infinity - Demucs + Whisper su RunPod Serverless (scheda video NVIDIA)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "demucs==4.0.1" "runpod>=1.6" soundfile "openai-whisper==20240930"

# scarica i modelli dentro l'immagine, così ogni avvio è più rapido
RUN python -c "from demucs.pretrained import get_model; [get_model(n) for n in ('htdemucs','htdemucs_6s','htdemucs_ft')]"
RUN python -c "import whisper; whisper.load_model('turbo', device='cpu')"

COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
