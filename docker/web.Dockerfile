# The Flask app, with TotalSegmentator for finding the liver
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch: the default build brings several GB of GPU libraries this app doesn't use
RUN pip install --no-cache-dir torch==2.15.0 torchvision --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY app.py .
COPY static/ static/
COPY templates/ templates/
# TotalSegmentator sends usage stats unless its config says otherwise.
# Its weights go in a volume (see docker-compose.yml), so they only download once.
COPY docker/totalsegmentator-config.json /opt/totalsegmentator/config.json
ENV TOTALSEG_HOME_DIR=/opt/totalsegmentator
ENV TOTALSEG_WEIGHTS_PATH=/weights
ENV HOST=0.0.0.0
CMD ["python", "app.py"]
