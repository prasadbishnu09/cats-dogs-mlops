FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# install python deps
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# copy source and model
COPY src/ src/
//COPY models/ models/

EXPOSE 8000

RUN useradd --create-home appuser
USER appuser

CMD ["uvicorn", "src.serve.main:app", "--host", "0.0.0.0", "--port", "8000"]