FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all gateway source files
COPY main.py state.py scheduler.py watcher.py ./

RUN useradd -m -u 1000 peag && chown -R peag:peag /app
USER peag

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
