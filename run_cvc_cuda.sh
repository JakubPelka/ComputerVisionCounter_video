#!/bin/bash

cd "$HOME/GitHub/ComputerVisionCounter_video" || exit 1

source .venv_cuda/bin/activate

# Wymusza RTSP over TCP dla OpenCV/FFmpeg.
# To ogranicza błędy H.264 typu:
# "error while decoding MB ..., bytestream ..."
export OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"

python src/start.py "$@"
