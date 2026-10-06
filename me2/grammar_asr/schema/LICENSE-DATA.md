# Dataset licensing notes (research / education only)

Primary shared dataset:
[course gold dataset](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands)
DOI `10.57967/hf/10723`.

There is **no single permissive license** for the combined release. Each source
keeps its own terms; the combined dataset is for **research and education only**
(non-commercial) under the most restrictive contributing terms.

| Source | Terms (summary) |
| --- | --- |
| SLURP | CC BY 4.0 |
| Google Speech Commands v2 | CC BY 4.0 |
| Common Voice 19 (en) | CC0 |
| Fluent Speech Commands | Non-commercial academic license |
| SNIPS SLU | See upstream SNIPS SLU terms |
| Timers and Such | See upstream license |
| MLEnd spoken numerals | See MLEnd terms |
| Multi-Sensor Voice Command | CC BY 4.0 (+ download-tracking obligation) |
| Group synthetic set | Class internal; LibriSpeech CC BY 4.0 refs |
| DEMAND noise | CC BY-SA 3.0 |
| MS-SNSD noise | MIT |
| supplemental_fil (CosyVoice2 clones) | Research/education only; reference consent and CosyVoice2 weight license **unverified** — treat as optional ablation only |
| Class recordings | Do not redistribute outside class without speaker consent |

Always record the Hugging Face revision / local fingerprint used for a run.
Do not train on official `test` or `holdout`. Use `tune_oos` only for
threshold selection, never for training.
