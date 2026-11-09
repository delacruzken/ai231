# Speech model weights (Phase 1)

Weights are **not** committed. Fetch with:

```bash
./scripts/fetch_speech_models.sh
```

Pinned artifacts (see `MANIFEST.json` after fetch):

| Role | Artifact |
|---|---|
| ASR | `sherpa-onnx-streaming-zipformer-en-2023-02-21` (INT8 encoder/joiner + decoder) |
| TTS | `en_US-lessac-medium.onnx` via **piper-tts** (sherpa-onnx 2.x OfflineTts phonemize path deprecated) |
| VAD | `silero_vad.onnx` (copied from `../fd-benchmark/models/`) |

Sample rates: ASR/VAD 16 kHz mono; Piper lessac-medium native **22050** Hz — resample only at a documented boundary in the runner.
