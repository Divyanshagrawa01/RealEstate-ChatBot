
FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY streamlit_app.py .
COPY data/ ./data/
COPY chroma_db/ ./chroma_db/

EXPOSE 8501

CMD ["sh", "-c", "streamlit run streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
