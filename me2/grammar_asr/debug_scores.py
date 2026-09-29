import sys
sys.path.insert(0, "/Users/hdc/sandbox/ai231/me2")
import torch
from grammar_asr.asr_model import CTCEncoder, greedy_decode
from grammar_asr.features import load_wav, log_mel, normalize_feat, load_manifest
from grammar_asr.grammar import build_grammar, NUM_CLASSES, idx_to_char
from grammar_asr.decode_grammar import score_all_words

device = "mps"
model = CTCEncoder().to(device)
model.load_state_dict(torch.load("grammar_asr/runs/base30/best.pt", map_location=device))
model.eval()
trie = build_grammar()

samples = [s for s in load_manifest(split="test", condition="clean") if s.intent == "ALARM"][:3]
for s in samples:
    audio = load_wav(s.path)
    feat = normalize_feat(log_mel(audio))
    x = torch.from_numpy(feat).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).squeeze(0)
    T = logits.shape[0]
    ids = greedy_decode(logits.unsqueeze(0))[0]
    gt = "".join(idx_to_char[i] for i in ids if i != NUM_CLASSES - 1)
    print(s.path.name)
    print(f"  T_frames={T}  ref={s.transcript!r}  greedy={gt!r}")
    log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
    res = score_all_words(log_probs, trie, frame_penalty=0.1)[:5]
    for r in res:
        print(f"    score={r.score:8.2f}  {r.intent:16s} {r.transcript!r}")
    print()
