# ME2 — Team Progress Synthesis
*Compiled Sep 24, 2026 from the two Telegram exports (`ai222/messages_222.html`, `ai231/messages_231.html`) and cross-referenced against `AI231 ME2 - Datasets.pdf` and `ME2-PLAN.md`.*

Two groups are working in parallel: **AI 222** (the earlier/seed group, Sep 13–14) and **AI 231** (the main group, Sep 14–24). Many members overlap. Below is what has been decided, what each person has done, and what is still open.

---

## 1. Key decisions already made (class-wide)

| # | Decision | Status | Source |
|---|----------|--------|--------|
| 1 | **Intent classification, not ASR.** The model outputs one of ~10 command classes (multiclass), not a transcription. "Don't complicate things" (Sir). | ✅ Agreed | 222 chat, King/Anthony/Mark |
| 2 | **Predefined actions per command.** "set alarm" → set a fixed time; "play music" → play a fixed MP3; "lights on/off" → toggle an LED. No free-form parameters, no numbers. | ✅ Agreed | 222 chat, Mark/King |
| 3 | **Continuous listening** (no mandatory "Hey Siri" wake word). Device listens continuously for commands. A wake word is *optional*, not required. | ✅ Clarified with Sir | 222 chat, King |
| 4 | **Custom wake word NOT required.** Teams may use an available one or skip it. | ✅ Agreed | 231 chat, Daniel/King |
| 5 | **Must respond to ANY person's voice** (not just the owner). Direct quote from Sir: *"You have the freedom to adjust the needed datasets. Bottom line, you have to train a model from scratch and should be able to respond to any person given the listed commands."* | ✅ Quoted | 231 chat |
| 6 | **Datasets are flexible — teams do NOT all have to use the same data.** Sir's only concern is server memory. Each team picks/generates what fits its objective. | ✅ Quoted | 231 chat, Ailene |
| 7 | **Training from scratch is mandatory** (no pretrained ASR models). Borrowing the *architecture/structure* of ASRs is OK. | ✅ Quoted | 231 chat |
| 8 | **Language: English assumed** (Sir never specified; team consensus). | ⚠️ Assumed | 222 chat |
| 9 | **Pool voices across the team** to cover "any person" — agreed not "hacky" because equipment is limited. | ✅ Agreed | 231 chat |
| 10 | **Benchmarks (still being finalized):** latency / inference time, efficiency, command recall (attempts until success), task-completion/success rate, possibly WER. **New proposal (Sep 24):** N external evaluators (not owners) each say each command N times; owner captures output logs for accurate time metrics. | 🔄 In progress | 231 chat, Ailene |

**Open question nobody has closed yet:** whether wake-word accuracy counts separately from command-mode accuracy (two different models → two different benchmark sets). Ailene raised this; no resolution recorded.

---

## 2. What each person has done so far

### Data generation (synthetic)
- **Anthony Navarez (Martinnavs)** — Most advanced on synthetic audio.
  - Generated synthetic commands over the weekend using **CosyVoice**; spot-checks look good.
  - Pipeline: 1 recorded prompt (e.g. "Play Music") + several reference audio files → CosyVoice generates outputs.
  - Data: Google Drive `1P2zogBUNzHAHW0IA3cNTFJgmwi8Ymlvo` (outputs in `conversions/`, raw in `cosy-voice-data/`).
  - Repo: `github.com/Martinnavs/AI-222-Machine-Exercises` (branch `ME2/ME2`).
  - Second repo `github.com/Martinnavs/simple-audio-transcriber` — transcribes generated audio and validates against filename (e.g. `Lights off.wav` → does it transcribe to "Lights off"), with a probability score + human-in-the-loop for low-confidence records. This automates generation **and** labeling.
  - Next plan: add noise + overlay background/home audio to increase input variety.
- **Mark Macalalad** —
  - Generated synthetic data with **Chatterbox TTS** using **LibriSpeech** recordings as reference voices + Filipino reference recordings from **SilencioNetwork / tagalog-filipino-speech** (notes some outputs sound Indian-accented).
  - Has ~**10k eSpeak NG** clips; planning ~**9k Chatterbox Nano** clips with speaker-disjoint splits (20 LS + 4 FE train; 2 LS + 1 FE val/test each).
  - Proposed the shared Google Drive for compiled datasets (organized by label): `1YWwWkP1L4MfnJ5rCzqI5fmNBwKht0NW-`.
  - Workaround for the "music playing lowers next-command confidence" edge case: while music plays, require the wake word and drop music volume to 5% while awaiting the next command (no extra noise augmentation needed — "goods naman").
- **Ailene Mondares** — Has **~56 GB of training data** (largest personal set). Training on her own device (HPC at 100% utilization, so she trains locally).
- **Unattributed member (222 chat)** — Generating synthetic data now: **20 labels × 30 speakers × 10 phrases × 3 variations** (clean / with-noise / far-from-mic), incl. 6 Filipino speakers; will share when done.

### Data collection / curation (real)
- **Quiel** — Matched the team's data schema to **SLURP** metadata. Findings: `sentences` = unique prompts (e.g. "wake me up at ten" → `alarm_set`); `recordings` = multiple audio per sentence (diff. speakers/mics); mapped SLURP intents → team labels mostly 1-to-1 (LIGHT_OFF = 2 legacy duplicates, BRIGHTNESS = 4 naming variants). Had not yet downloaded all audio at last message.
- **King** — Preliminary results on a voice-command-specific dataset; requested reference links from peers.
- **Daniel Nepomuceno** — Searching for existing STT/voice-command corpora; suggested ETL workflow; found a Kaggle notebook (GSC-based, with VAD).
- **Joven Daniel Nepomuceno** — Collected the Kaggle synthetic-commands dataset (in the dataset sheet).

### Hardware / device
- **LAPDG (Loreen?)** — Ordered the RPi 5 (8 GB) from DigiKey (~₱10,788, no shipping); confirmed no extra custom tax. Started on datasets while waiting for delivery.
- **Ailene Mondares** — Bought a **Bluetooth smart bulb** + will bring a socket on demo day.
- **Cherry Magdaong** — Also bought a **smart bulb** (photos shared Sep 23); not yet tested on RPi.
- **Kent Justin Canja** — Plans to **simulate lights via laptop screen** (no physical hardware).
- **Renz** — Built an initial capture script: press space → record 500 ms clip.
- Multiple members confirmed their smart bulbs currently work **via phone only**; RPi connectivity not yet tested.

### Coordination / meta
- **Cherry Magdaong** — Created the 231 supergroup + topics (Resources, Announcements, Poll); maintains the master **Google Sheet** of found/generated datasets (`1VFm1-SAdNqtSwOeSHZct23tF6pPTntmj2940sugj61E`); created a separate 231 GC because not all 222 members are in 231.
- **Ailene Mondares** — Driving the benchmark discussion; proposed the N-evaluator protocol; asking Sir about borrowing ASR architecture.
- **LAPDG** — Shared Sir's DL repo: `github.com/roatienza/Deep-Learning-Experiments`.

---

## 3. Shared resources (links)

| Resource | Link | Owner |
|---|---|---|
| Master dataset spreadsheet | `docs.google.com/spreadsheets/d/1VFm1-SAdNqtSwOeSHZct23tF6pPTntmj2940sugj61E` | Cherry |
| Compiled dataset Drive (by label) | `drive.google.com/drive/folders/1YWwWkP1L4MfnJ5rCzqI5fmNBwKht0NW-` | Mark |
| Anthony's synthetic data | `drive.google.com/drive/folders/1P2zogBUNzHAHW0IA3cNTFJgmwi8Ymlvo` | Anthony |
| Anthony's generation repo | `github.com/Martinnavs/AI-222-Machine-Exercises` (ME2 branch) | Anthony |
| Anthony's transcriber | `github.com/Martinnavs/simple-audio-transcriber` | Anthony |
| Sir's DL experiments | `github.com/roatienza/Deep-Learning-Experiments` | LAPDG |
| 231 group invite | `t.me/+J8L4xwwBTHRjODY9` | Cherry |

---

## 4. Gap analysis — what's done vs. what ME2 still requires

### ✅ Done / in flight
- Problem framed (intent classification, predefined actions, continuous listen, any-person).
- Substantial synthetic data generated (Anthony CosyVoice, Mark Chatterbox/eSpeak, Ailene 56 GB, unattributed 20-label set).
- Real-data mapping underway (Quiel→SLURP, King, Daniel, Joven).
- Hardware procured for several members (RPi 5, smart bulbs).
- Centralized tracking (spreadsheet + Drive).

### ⏳ Still to do (to meet the spec)
1. **Finalize the unified label schema** — the team referenced an "Option B" schema sheet but no final 10-label list is locked in the chats. *(Your `ME2-PLAN.md` proposes one — push it to the group.)*
2. **Lock the benchmark protocol** — the N-evaluator design is proposed but unconfirmed; decide whether wake-word accuracy is scored separately.
3. **Download + merge the real datasets** — Quiel hadn't finished SLURP audio; nobody has confirmed a merged `train/val/test` tree with a `labels.csv`.
4. **Confirm SynTTS full-download access** — the HF link is masked for review (Cherry owns it).
5. **Train the model from scratch** — only data work is visible in the chats; no one has posted a trained model or accuracy yet. (Ailene mentioned her machine is slow / HPC saturated.)
6. **Deploy to RPi 5** — bulbs work via phone; **no RPi connectivity tested yet**. This is the biggest unknown.
7. **Edge-case handling** — music-playing → next-command confidence drop (Mark's 5%-volume + wake-word workaround is a candidate; not validated).
8. **Output logging** — Ailene noted models must emit logs for the time metrics; no logging spec yet.
9. **Demo assets** — 2-min demo video, report, and the actual action dispatch (MP3 playback, LED toggle, alarm set) on-device.

### ⚠️ Risks / watch-outs
- **"Any person" requirement** is the hardest constraint — pooling team voices + synthetic multi-speaker data is the mitigation, but it must be explicit in the test set (speaker-disjoint).
- **Server memory** is Sir's stated concern for pooled data — keep the shared Drive lean.
- **Synthetic accent leakage** (Mark's Indian-accent note) — may hurt Filipino-user realism; consider weighting real/FE-synthetic data.
- **Timeline:** deadline Sep 26, 1:59 PM (~2 days from the chat's last message). Training + RPi deployment are the critical path and neither is started in the chats.

---

## 5. Suggested next actions for you (Rowel)
1. Post your `ME2-PLAN.md` label table + architecture (MicroCNN/TC-ResNet) to the 231 group to close decision #1 and the schema gap.
2. Confirm the benchmark protocol (N evaluators × N repeats + owner logs) in the Poll topic.
3. Assign dataset merges: Quiel finishes SLURP, Mark finalizes the Drive tree, Cherry unblocks SynTTS.
4. Prioritize **RPi 5 + mic + bulb connectivity test** this week — it's the least-proven piece and gates the demo.
5. Stand up output logging early so benchmark time metrics are capturable from the first training run.
