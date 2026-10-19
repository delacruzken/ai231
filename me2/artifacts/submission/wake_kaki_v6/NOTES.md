# wake_kaki_v6

Retrain after USB-mic (`USB PnP Sound Device`, ffmpeg `:0`) positives + hard negatives.

- Positives: `data/wake/kaki/positive` (**98** total; +20 `hdc_usb_*`)
- Negatives: existing hard/general + mac ambient + **48** `hdc_usb_hard_*` + gold train audio
- Train: 50 epochs, MPS, seed **5**, target FAR 0.05
  - Seed **4** (same recipe as v5) collapsed to recall ~60% — discarded (`wake_kaki_v6b` was the seed-5 trial, then promoted here)
- Best val: balanced_acc **98.5%**, recall **100%**, FAR **3.0%**, threshold **0.637**
- Live start thr: **0.55** in `live_debug_60s.sh` (val thr 0.637; lower for USB reactivity)
- Artifacts: `best.pt`, `kaki.onnx`, `kaki_int8.onnx`
- Command model unchanged: `intent_hf_v1`
