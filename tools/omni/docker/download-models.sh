#!/usr/bin/env bash
# Downloads what llama-omni-server needs into $OMNI_MODELS (default /models):
#   MiniCPM-o-4_5-gguf/  LLM (QUANT, default Q4_K_M), audio/, tts/, token2wav-gguf/
#                        (+ vision/ with --vision; audio-only sessions skip it)
#   voice-clone/         CAM++ and the speech tokenizer for voice cloning
# Resumable; files already complete are skipped. HF_ENDPOINT picks a mirror.
set -euo pipefail

MODELS="${OMNI_MODELS:-/models}"
QUANT="${QUANT:-Q4_K_M}"
HF="${HF_ENDPOINT:-https://huggingface.co}"
VISION=0
[ "${1:-}" = "--vision" ] && VISION=1

fetch() {  # fetch <repo> <path in repo> <destination>
  local url="$HF/$1/resolve/main/$2" dest="$3" size have i
  mkdir -p "$(dirname "$dest")"
  size="$(curl -sIL --max-time 60 "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')"
  for i in $(seq 1 20); do
    have="$(stat -c %s "$dest" 2>/dev/null || echo 0)"
    if [ -n "$size" ] && [ "$have" = "$size" ]; then
      echo "ok   $dest"
      return 0
    fi
    echo "get  $dest ($have/${size:-?} bytes, try $i)"
    curl -fL --retry 3 --speed-time 30 --speed-limit 10000 -C - -o "$dest" "$url" || sleep 3
  done
  echo "FAILED $dest" >&2
  return 1
}

GGUF=openbmb/MiniCPM-o-4_5-gguf
D="$MODELS/MiniCPM-o-4_5-gguf"
fetch $GGUF "MiniCPM-o-4_5-$QUANT.gguf" "$D/MiniCPM-o-4_5-$QUANT.gguf"
fetch $GGUF audio/MiniCPM-o-4_5-audio-F16.gguf "$D/audio/MiniCPM-o-4_5-audio-F16.gguf"
fetch $GGUF tts/MiniCPM-o-4_5-tts-F16.gguf "$D/tts/MiniCPM-o-4_5-tts-F16.gguf"
fetch $GGUF tts/MiniCPM-o-4_5-projector-F16.gguf "$D/tts/MiniCPM-o-4_5-projector-F16.gguf"
for f in encoder flow_matching flow_extra hifigan2 prompt_cache; do
  fetch $GGUF "token2wav-gguf/$f.gguf" "$D/token2wav-gguf/$f.gguf"
done
if [ "$VISION" = 1 ]; then
  fetch $GGUF vision/MiniCPM-o-4_5-vision-F16.gguf "$D/vision/MiniCPM-o-4_5-vision-F16.gguf"
fi
fetch openbmb/MiniCPM-o-4_5 assets/token2wav/campplus.onnx "$MODELS/voice-clone/campplus.onnx"
fetch openbmb/MiniCPM-o-4_5 assets/token2wav/speech_tokenizer_v2_25hz.onnx \
  "$MODELS/voice-clone/speech_tokenizer_v2_25hz.onnx"
echo "done: $MODELS"
