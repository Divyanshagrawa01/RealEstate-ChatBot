
FROM python:3.13-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY main.py .
COPY streamlit_app.py .
COPY data/ ./data/

# Create ChromaDB directory
RUN mkdir -p /app/chroma_db

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]