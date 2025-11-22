FROM python:3.10-slim

# Instala dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copia o código
COPY . /app
WORKDIR /app

RUN ls -lh /app/model/

# Inicia o serviço (PORT dinâmica do Render)
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
