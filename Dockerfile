# Gradio TWA laboratory (app.py). The image installs only what the lab needs:
# the cardioonco library (numpy, scipy) plus gradio and matplotlib -- no torch.
FROM python:3.12-slim

WORKDIR /code
COPY pyproject.toml README.md LICENSE ./
COPY cardioonco ./cardioonco
RUN pip install --no-cache-dir ".[app]"
COPY app.py ./

EXPOSE 7860
CMD ["python", "app.py"]
