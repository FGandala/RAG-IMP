FROM python:3.11-slim as builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instala compiladores necessários (gcc) para algumas libs de IA
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala requirements
COPY requirements.txt .
# Instala em uma pasta local (/install) para podermos copiar depois
RUN pip install --prefix=/install --no-warn-script-location -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

# Cria usuário não-root 
RUN addgroup --system appgroup && adduser --system --group appuser

# Copia as bibliotecas instaladas do estágio anterior
COPY --from=builder /install /usr/local

# Copia o código fonte
COPY ./app ./app

# Define variáveis de ambiente para o Hugging Face 
ENV HF_HOME=/app/data/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/data/.cache/huggingface

# Cria a pasta de dados e dá permissão para o usuário 
RUN mkdir -p /app/data && chown -R appuser:appgroup /app

# Muda para o usuário seguro
USER appuser


EXPOSE 8080

# Comando de Start
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}