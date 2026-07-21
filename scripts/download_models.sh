#!/usr/bin/env bash
# Fetch the OpenCV Zoo face models that enable production identity matching.
# Without these the app runs in demo mode: liveness works, but face match
# similarity is indicative only and identity is never confirmed.
set -euo pipefail
DIR="$(dirname "$0")/../models"
mkdir -p "$DIR"
BASE="https://github.com/opencv/opencv_zoo/raw/main/models"
echo "Downloading YuNet face detector..."
curl -fL -o "$DIR/face_detection_yunet_2023mar.onnx" \
  "$BASE/face_detection_yunet/face_detection_yunet_2023mar.onnx"
echo "Downloading SFace face recognizer..."
curl -fL -o "$DIR/face_recognition_sface_2021dec.onnx" \
  "$BASE/face_recognition_sface/face_recognition_sface_2021dec.onnx"
echo "Done. Restart the server to activate the production face engine."
