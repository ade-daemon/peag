FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=120 -r requirements.txt

COPY main.py state.py scheduler.py watcher.py task_intelligence.py agent_executor.py agent_scheduler.py agent_builder.py tool_registry.py ./

RUN useradd -m -u 1000 peag && chown -R peag:peag /app
USER peag

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
