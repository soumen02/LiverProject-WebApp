# Running this project

The code in this repo up to commit `21fe4ba` (July 2023) was written by hand. The changes after that
(November 2026) were made with Claude Code, only to get the project running again. They are listed below.

## With Docker

This is the easiest way. It needs Docker with about 16 GB of memory.

```bash
git clone --recursive https://github.com/soumen02/LiverProject-WebApp.git
cd LiverProject-WebApp
docker compose up --build
```

Then open http://127.0.0.1:5055. The rest of this page covers running it without Docker.

## Start the model servers

The trained models are in the `LiverProject-noGUI` submodule. Set it up as described in its `RUNNING.md`,
then start one server per phase from inside it:

```bash
python -m monailabel.main start_server --app radiologyPV --studies StudiesPV --conf models pv_segmentation --conf use_pretrained_model false --port 8000
python -m monailabel.main start_server --app radiologyHA --studies StudiesHA --conf models ha_segmentation --conf use_pretrained_model false --port 8001
python -m monailabel.main start_server --app radiologyHV --studies StudiesHV --conf models hv_segmentation --conf use_pretrained_model false --port 8002
```

## Start the web app

In a separate Python 3.11 environment:

```bash
pip install -r requirements.txt
PORT=5055 python app.py
```

Open http://127.0.0.1:5055 and upload a contrast CT as `.nii` or `.nii.gz`.
Uploading a single `.dcm` only converts that one slice, so it is not enough for a real CT series.

This is a 2023 research prototype. It has not been clinically validated, and it should stay on localhost.

## What changed in November 2026

- Pointed the app at the models in LiverProject-noGUI. The old `*_livercrop_unet` names are not in either repo.
- The Venous Phase button now uses the hepatic vein model. It was calling the portal vein model.
- The liver bounding box now includes its last slice in each direction.
- Replaced `fslmaths -mas` with the same masking in nibabel, so FSL is no longer required.
- Failed steps now return an error. A failed inference used to be saved as a `.nii.gz` download.
- Added `requirements.txt`, turned off debug mode, stopped hard-coding the secret key, and removed a committed `.pyc`.
- Added the Docker setup, a sample CT, and `LiverProject-noGUI` as a submodule.
- The Segment button now shows that a scan is being processed.
- TotalSegmentator's usage stats are turned off in the Docker image.
- Rewrote the README with the paper, poster and results. The paper's copy here has one case label removed from Figure 4.
