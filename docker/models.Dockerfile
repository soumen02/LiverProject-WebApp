# One image for the three MONAI Label servers (portal vein, hepatic artery, hepatic vein)
FROM python:3.9-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /models
COPY LiverProject-noGUI/requirements.txt .
RUN pip install --no-cache-dir "setuptools<70" wheel \
 && pip install --no-cache-dir torch==1.13.0 --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY LiverProject-noGUI/ .
