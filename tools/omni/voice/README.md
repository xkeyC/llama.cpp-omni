# Voice assistant features (AstrBot)

Additions to `llama-omni-server`'s WebSocket `/backend` for running MiniCPM-o 4.5
as the voice of a chat bot whose real work is done by an agent elsewhere
(AstrBot + Codex). All of them are opt-in: without the settings below the
server behaves as upstream.

## Running on a 12 GB GPU

```bash
export OMNI_TTS_N_CTX=4096     # TTS context of its own (it was n_ctx; the TTS is trained on 4096)
export OMNI_ROUTER_CTX=2048    # enables the tool router (extra KV cells for its sequences)
export OMNI_VOICE_BUNDLE_CMD='python tools/omni/voice/make_voice_bundle.py --models <dir>'
llama-omni-server -m MiniCPM-o-4_5-Q4_K_M.gguf -ngl 99 -c 8192 -fa on -ctk q8_0 -ctv q8_0 \
  --temp 0.7 --top-k 100 --top-p 0.8 --repeat-penalty 1.05 --min-p 0 --host 127.0.0.1 --port 19060
```

Measured on an RTX 4090 (whole-card memory increase, audio-only session):

| setup | VRAM |
|---|---|
| upstream defaults (vision loaded, 8K, f16 KV) | 11.7 GB |
| `vision: false` | 10.7 GB |
| + `OMNI_TTS_N_CTX=4096` | 10.3 GB |
| + q8_0 KV, flash attention | **9.7 GB** |
| same with `-c 16384` | 10.2 GB |

Keep `-c` at 8192: that is the context the model was trained on; beyond it
answers degrade (it stops answering after ~8K tokens in long sessions). The
duplex sliding window keeps the session going past it.

`--host` is honoured now (upstream always bound 0.0.0.0): the default is
127.0.0.1, so pass `--host 0.0.0.0` for a server reached from other machines
or containers (there is no authentication).

## Docker (CUDA)

`tools/omni/docker` builds `llama-omni-server` with CUDA from this checkout
and runs it with the settings above:

```bash
cd tools/omni/docker
docker compose build                    # CUDA_DOCKER_ARCH=86 builds for a 3060 only (faster)
docker compose run --rm omni download   # models into ./models; --vision adds the vision encoder
docker compose up -d                    # ws://127.0.0.1:19060/backend
```

- Needs the NVIDIA container toolkit (Docker Desktop with WSL 2 on Windows).
  The default build covers GPU architectures 75, 80, 86, 89, 90 and 120.
- Models live in `./models` (~9 GB: Q4_K_M LLM, audio, TTS, token2wav, the two
  voice-clone ONNX models); `QUANT=Q8_0` picks another LLM quantization and
  `HF_ENDPOINT` a mirror. Voice bundles are cached in `./cache`.
- Voice cloning is set up when `models/voice-clone` is there.
- On Windows keep the models on the WSL 2 file system or in a Docker volume:
  a bind mount of a Windows drive loads them several times slower (~8 min).
- `OMNI_CTX`, `OMNI_PORT` and `OMNI_EXTRA_ARGS` (more server flags) can be set
  in the environment or a `.env` file.
- The port is bound to loopback because the server has no authentication;
  change the mapping in `docker-compose.yml` only on a trusted network. A bot
  in another container reaches it without any port mapping by joining the
  compose network: `ws://omni:19060/backend`.
- Build arguments take `HTTP_PROXY` / `HTTPS_PROXY` from the environment.

## session.init

- `vision: false` — audio-only session; the vision encoder is not loaded.
- `system_prompt` — plain text is placed inside the duplex system turn, after
  the reference audio (text starting with `<|` is used verbatim, as before).
- `voice.ref_audio` — also sets token2wav's voice (voice clone), not only the
  LLM's: a prompt bundle is built once per recording by `OMNI_VOICE_BUNDLE_CMD`
  (`make_voice_bundle.py <wav> <out_dir>`, needs `campplus.onnx` and
  `speech_tokenizer_v2_25hz.onnx` from the MiniCPM-o 4.5 repo) and cached in
  `OMNI_VOICE_CACHE_DIR` (default `<temp>/omni_ws/voices`). Without reference
  audio the default voice is restored. The models stay loaded. Without
  `OMNI_VOICE_BUNDLE_CMD` the reference audio only styles the LLM prompt, as
  upstream, and token2wav keeps its default voice.
- `config.say_tokens_per_chunk` — forced speech rate (default 4 tokens per
  unit, about 1 s of speech; the TTS makes at most ~1 s of audio per unit).
- `config.router` — the tool router (below).

## input.append (full duplex)

- `say` — forced speech: from the next unit where the model is not speaking,
  the units speak this text verbatim as the model's own turn (it stays in the
  LLM context and goes through the TTS with the session voice).
- `say_cancel` — drops forced speech not spoken yet.
- `voiced` — the client's VAD: whether this unit has speech (otherwise RMS).
- `transcript` — the client ends an utterance with this unit and says what it
  was; the router uses it instead of transcribing.

## Tool router

The duplex model never calls tools and does not follow instructions such as
"stay silent unless addressed". The same LLM does in text mode, so every user
utterance (a run of voiced units) is decided on extra sequences of the same
context, with no extra weights: transcribed (sequence 2, unless the client
sent a transcript), then a tool is chosen for the transcript with a Qwen3
tools prompt (sequence 1, system prompt cached):

```json
"router": {
  "system": "<tools prompt with <tools>...</tools>>",
  "tools": ["silence", "reply", "backend_task"],
  "bias": {"silence": 4.0},
  "user_template": "{heard}",
  "transcribe_prompt": "请仔细听这段音频片段，并将其内容逐字记录。",
  "audio_units": 12, "silence_hold": 1, "tool_hold": 3,
  "client_transcripts": true
}
```

The tool name is chosen greedily among the configured names (with the logit
bias); arguments are generated for other tools. With `client_transcripts`, an
utterance ends when its transcript arrives (not when the client marks the
voice as over), so each utterance is decided once. The tools prompt must fit
`OMNI_ROUTER_CTX` with ~1024 tokens to spare, or the router turns itself off.
Every `session.init` configures the router anew (none without `config.router`). Decisions come back as

```json
{"type": "response.tool_call", "name": "backend_task", "arguments": {"task": "..."},
 "heard": "...", "interrupted": true}
```

- `reply`: the model answers: in that unit, or when the speaker stops if
  they went on talking (a model already speaking just goes on);
- `silence`: the model keeps listening (and stops if it was speaking);
- any other tool: reported to the client, the model keeps listening; the
  client typically runs the task and sends the result as `say`.

The model does not start speaking while an utterance is going on, and after
answering and yielding it waits for the next utterance: left alone it keeps
greeting and repeats itself. (A `reply` decided while it was already
speaking still allows one turn after it yields.) `interrupted` says the decision stopped the
model's own speech.
