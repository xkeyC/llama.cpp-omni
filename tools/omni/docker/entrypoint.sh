#!/usr/bin/env bash
# serve (default): run llama-omni-server; extra arguments are passed on.
# download: fetch the models into $OMNI_MODELS (see download-models.sh).
# Anything else is run as a command.
set -euo pipefail

cmd="${1:-serve}"
case "$cmd" in
  download) shift; exec /app/download-models.sh "$@" ;;
  serve) [ $# -gt 0 ] && shift ;;
  *) exec "$@" ;;
esac

MODELS="${OMNI_MODELS:-/models}"
LLM="${OMNI_LLM:-$MODELS/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf}"
if [ ! -f "$LLM" ]; then
  echo "Model not found: $LLM" >&2
  echo "Download it first: docker compose run --rm omni download" >&2
  exit 1
fi

# Voice clone needs the two ONNX models; without them the reference audio
# only styles the LLM prompt and the default voice is used.
if [ -z "${OMNI_VOICE_BUNDLE_CMD+x}" ] && [ -f "$MODELS/voice-clone/campplus.onnx" ]; then
  export OMNI_VOICE_BUNDLE_CMD="python3 /app/tools/omni/voice/make_voice_bundle.py --models $MODELS/voice-clone"
fi
mkdir -p "${OMNI_VOICE_CACHE_DIR:-/cache/voices}"

# Line-buffered: the server logs with printf, which a pipe would hold back.
# shellcheck disable=SC2086 # OMNI_EXTRA_ARGS is split on purpose
exec stdbuf -oL -eL /app/llama-omni-server -m "$LLM" -ngl 99 -c "${OMNI_CTX:-8192}" \
  -fa on -ctk q8_0 -ctv q8_0 \
  --temp 0.7 --top-k 100 --top-p 0.8 --repeat-penalty 1.05 --min-p 0 \
  --host 0.0.0.0 --port "${OMNI_PORT:-19060}" ${OMNI_EXTRA_ARGS:-} "$@"
