FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libsamplerate0-dev \
    libtag1-dev \
    libchromaprint-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /app/models && \
    curl -L -o /app/models/audioset-vggish-3.pb \
    https://essentia.upf.edu/models/feature-extractors/vggish/audioset-vggish-3.pb && \
    curl -L -o /app/models/voice_instrumental-audioset-vggish-1.pb \
    https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-audioset-vggish-1.pb

RUN pip install --no-cache-dir \
    essentia-tensorflow \
    numpy \
    mutagen

COPY tagger.py .

# This environment variable forces Python to flush stdout immediately 
# so you see status prints instantly!
ENV PYTHONUNBUFFERED=1

CMD ["python", "tagger.py"]
