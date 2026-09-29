# ME2 — Voice Command Model: Step-by-Step Execution Plan

**Hardware:** Raspberry Pi 5 (8 GB RAM) · **Deadline:** Sep 26, 1:59 PM (~2 days)
**Constraints:** Tiny model · real-time on RPi5 · fully on-device · no LLM · no cloud calls
**Scope:** Classify 10 smart-home voice commands (intent classification, not ASR)

---

## Architecture Decision (make this first, Day 1 morning)

**Recommended: MicroCNN / TC-ResNet over log-mel spectrograms**

Why:
- Proven at <1 ms inference on RPi-class hardware (SynTTS-Commands repo ships MicroCNN + TC-ResNet baselines with 93% acc on GSC)
- ~1–5 MB model size (vs. Whisper-small at 250 MB — disqualifies on "tiny")
- No LLM, no cloud — pure signal→intent
- Training is fast even on CPU (hours, not days)

Alternatives considered:
- YAMNet (12 M params) — works but heavier than needed
- Distilled transformer — overkill; no accuracy gain worth the latency on RPi5
- Keyword-spotting (openWakeWord/Porcupine) — too rigid for 10 multi-word intents

**Pipeline (all on-device):**
```
mic (16 kHz mono) → 30-frame buffer → VAD gate (webrtcvad, ~0.1 ms)
  → log-mel spectrogram (n_mels=40, 25 ms hop) → MicroCNN/TC-ResNet
  → softmax over 10 intents (+ optional "silence/unknown" class)
  → confidence threshold → action dispatch
```

---

## Day 0 — Tonight (Sep 24 evening): Setup & Data Assembly

### 0.1 RPi5 preparation (do this first — it gates everything)
1. Flash latest Raspberry Pi OS Lite (64-bit) to SD/eMMC
2. `raspi-config`: enable audio (USB mic or ReSpeaker HAT), set hostname, enable SSH
3. Install: `python3.11`, `pip`, `numpy`, `sounddevice`, `torchaudio` or `librosa`, `onnxruntime`
4. Verify mic capture: `arecord -d 3 -f S16_LE -r 16000 -t wav test.wav && aplay test.wav`
5. Baseline: time a 40×T log-mel + small CNN forward pass → confirm <10 ms/frame budget

### 0.2 Assemble the dataset (collective task — divide with team)

**Unified label scheme (10 classes + 1 reject):**
| Label | ME2 command | Primary sources |
|---|---|---|
| PLAY_MUSIC | 1. Play music | Fluent Speech Commands, SLURP, SynTTS (cmd 1) |
| ASK_QUERY | 2. Ask question/search | Common Voice (filtered), SLURP |
| LIGHT_ON_OFF | 3. Control lights | Fluent Speech Commands, Snips SLU (smart-lights), GSC v2 "on"/"off" |
| LIGHT_DIM_COLOR | 4. Dim/color lights | Fluent Speech Commands (percent/color slots) |
| SET_TIMER | 5. Set a timer | **Timers and Such (real)**, Fluent, eSpeak synth |
| SET_ALARM | 6. Set an alarm | **Timers and Such (real)**, Fluent, eSpeak synth |
| THERMOSTAT | 7. Adjust thermostat | Fluent (temp phrases), eSpeak/Chatterbox synth |
| MEDIA_CTRL | 8. Pause/stop/next/volume | **SynTTS (cmd 8, 53k)**, GSC v2 "stop", Fluent |
| REMINDER_LIST | 9. Reminders/lists | Snips SLU, SLURP |
| CALL_MSG | 10. Calls/messaging | Snips SLU (partial), targeted synth |
| UNKNOWN | reject / not-a-command | GSC v2 background noise, ambient recordings |

**Per-source integration actions:**
1. **Timers and Such** (Quiel's filtered subset) → SET_TIMER / SET_ALARM. Gold real data.
2. **Fluent Speech Commands** (97 spk × 248 phr → 31 intents) → map 31 intents → our 10 labels via the slot table (action/object/location). Covers 1, 3, 4, 5, 6, 7, 8.
3. **Snips SLU v1.0** (5,890 rows, HF) → LIGHT_ON_OFF, REMINDER_LIST, CALL_MSG. Parse `intent` field; discard entities for classification.
4. **SLURP** (~6 GB audio on Zenodo) → select utterances matching our 10 labels via text annotation. CC BY-NC (fine for coursework).
5. **Google Speech Commands v2** → "stop", "on", "off", "up", "down" → MEDIA_CTRL / LIGHT_ON_OFF; `_background_noise_` → UNKNOWN class.
6. **SynTTS-Commands** (Cherry) → 14k cmd-1 + 53k cmd-8 (cosyvoice2). ⚠️ Full download is masked for review — confirm with Cherry whether HF link is unmasked; otherwise use samples zip.
7. **eSpeak NG** (Mark's 10k) → fill gaps in THERMOSTAT, LIGHT_DIM_COLOR, CALL_MSG (thinly covered).
8. **Chatterbox Nano** (Mark's planned 9k) → generate with speaker-disjoint splits (20 LibriSpeech + 4 FE train; 2 LS + 1 FE val/test each). Use Nano 110M (CPU, 3× realtime) for batch gen.
9. **Common Voice** (King) → ASK_QUERY class + Filipino-accent robustness. Filter by transcript keywords ("what", "who", "weather", "time").
10. **Kaggle synthetic-commands** (Joven) → prune to "on", "off", "stop" → augment LIGHT_ON_OFF / MEDIA_CTRL.

**Blend strategy (target ~50–80k total clips, ~1–3 s each):**
- Real speech: ~30–40% (Timers and Such, Fluent, Snips, SLURP, GSC, Common Voice)
- Synthetic: ~60–70% (eSpeak 10k, Chatterbox 9k, SynTTS 67k, Kaggle synth)
- Per-class balance: aim ≥3,000 clips/class; undersampled classes (CALL_MSG, ASK_QUERY) get extra synthetic generation
- Speaker-disjoint test set: hold out all clips from 5+ speakers (2 LS + 1 FE per Mark's plan) for the test split; val from another 3 speakers

**Deliverable:** `data/` tree with `train/`, `val/`, `test/` subfolders, WAV 16 kHz mono, `labels.csv` (path, label, source, speaker_id, is_synthetic). Plus `label_map.json` documenting source-intent → ME2-label mappings.

---

## Day 1 (Sep 25): Train, Benchmark, Deploy

### 1.1 Train the VCM (individual task)
- Framework: PyTorch (CPU training is fine for a 1–5 M param CNN; ~1–3 h)
- Input: log-mel (n_mels=40, sr=16k, 25 ms window/hop) → shape (40, T≈120)
- Models to try (in order of preference):
  1. **MicroCNN** (SynTTS repo baseline) — ~0.5 M params
  2. **TC-ResNet** (SynTTS repo baseline) — ~2 M params
  3. Custom shallow CNN (3 conv layers + 2 FC) if above underperform
- Augmentation: SpecAugment (freq/time masking), random gain ±6 dB, background-noise mixing (GSC noise) at SNR 10–20 dB
- Loss: CrossEntropy (class weights if imbalanced); early stopping on val loss
- Track: per-class precision/recall/F1, macro-F1, confusion matrix, latency (ms/clip)

### 1.2 Benchmark design (collective task — agree on protocol today)
**Benchmark = 3-tier evaluation:**
1. **Clean tier:** speaker-disjoint test set, quiet room → headline accuracy
2. **Noise tier:** same clips + babble/music/pink noise at SNR 5/10/15 dB → robustness curve
3. **Latency tier:** on RPi5, measure end-to-end (mic→decision) p50/p95 in ms; target **<200 ms** for real-time feel
4. **Reject tier:** UNKNOWN-class recall (how often non-commands are correctly ignored) — critical for a hands-free device

Report format: table of (tier, accuracy, macro-F1, per-class F1, p50/p95 latency, model size MB). Compare at least 2 model sizes.

### 1.3 Validate (individual task)
- Run benchmark on final model; log all metrics
- Error analysis: top-5 confused pairs, worst-performing class, synthetic-vs-real accuracy gap
- Write 1-page validation summary with confusion matrix plot

### 1.4 Deploy to RPi5 (individual tasks 5–8)
1. Export trained model → **ONNX** (opset 13) or **TFLite** (int8 quantized)
2. On RPi5: `onnxruntime` (or TFLite) + `sounddevice` for mic stream
3. Inference loop:
   ```python
   while True:
       frame = stream.read(480)          # 30 ms @ 16 kHz
       if vad.is_speech(frame):
           buf.append(frame)
           if len(buf) > MAX_FRAMES: break
       else:
           if buf: predict_and_dispatch(buf); buf.clear()
   ```
4. Dispatch: map predicted intent → shell command / MQTT publish / GPIO toggle (lights demo)
5. Verify: speak all 10 commands aloud, confirm correct action + <200 ms response
6. Stress test: 10-min continuous run, monitor `vcgencmd measure_temp` + RAM (should stay <500 MB)

### 1.5 Demo prep (individual task)
- Physical setup: RPi5 + USB mic + LED strip (WS2812) or smart plug (for light on/off)
- Demo script: show 5 of the 10 commands live (lights on/off, dim, timer, play music → MP3 via `mpg123`, volume)
- Fallback: if a command misfires, show the confidence score on a serial console (transparency)

---

## Day 2 (Sep 26): Finalize & Submit (due 1:59 PM)

### Morning
- [ ] Final benchmark run (clean + noise + latency tiers) — lock numbers
- [ ] Write report: architecture, dataset composition (per-source counts), training config, benchmark results, error analysis, deployment latency, demo video link
- [ ] Record 2-min demo video (all 10 commands, or 5 live + 5 screen-cast)
- [ ] Package: model file (.onnx), inference script, README (setup + run instructions), benchmark report

### Afternoon (submit before 1:59 PM)
- [ ] Upload deliverables to LMS
- [ ] Team sync: merge collective-task contributions (dataset sheet finalized, benchmark protocol agreed)
- [ ] Buffer: 30 min before deadline for upload issues

---

## Risk Register & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SynTTS full download still masked | Medium | Lose 67k cmd-1/8 clips | eSpeak + Chatterbox synth covers those classes; confirm with Cherry tonight |
| ASK_QUERY / CALL_MSG underrepresented | High | Poor recall on classes 2 & 10 | Targeted Chatterbox generation (100+ phrases each); accept lower F1 and document |
| RPi5 audio driver issues | Low | Blocks demo | Test `arecord`/`aplay` tonight (step 0.1.4); fallback: USB soundcard |
| Training doesn't converge in 3 h | Low | Delays deploy | Start with pretrained-on-GSC weights; reduce to 40×120 input; CPU is fine for <5 M params |
| Latency >200 ms on RPi5 | Low | Fails "real-time" requirement | int8 quantization; reduce n_mels to 32; shorter window (20 ms) |
| Team coordination (collective tasks) | Medium | Dataset/benchmark incomplete | Agree label map + benchmark protocol tonight; each person owns their sources |

---

## Division of Labor (suggested, based on dataset ownership)

| Person | Collective task 1 (dataset) | Individual tasks |
|---|---|---|
| Mark Andrian | SLURP, Fluent, GSC v2, eSpeak 10k, Chatterbox 9k plan | Own VCM training + deploy (largest dataset contributor) |
| Joven Daniel | GSC v0.02, Kaggle synth, Fluent corpus (Kaggle) | Benchmark design lead (collective) |
| Quiel Quiwa | Timers and Such (filtered), Snips SLU | Validation + error analysis |
| Cherry Magdaong | SynTTS-Commands (confirm access) | Noise-robustness benchmark tier |
| King Villasin | Common Voice (ASK_QUERY + FE accent) | Demo build (RPi5 + LED strip) |

*(Adjust as the team sees fit — the key is that each person's individual tasks align with the data they already assembled.)*

---

## Immediate Next Actions (tonight)

1. **Flash RPi5 + verify mic** (30 min) — unblocks everything
2. **Team huddle (30 min):** agree on the 10-label scheme above + benchmark protocol
3. **Start data downloads** (Timers and Such subset, Snips SLU from HF, GSC v2 via TF Datasets — all <1 GB each)
4. **Confirm SynTTS access** with Cherry
5. **Begin eSpeak generation** if Mark's 10k isn't already done (script is straightforward: espeak-ng CLI × 20 labels × 10 voices × 10 phrases × 5 speed/pitch variants)
