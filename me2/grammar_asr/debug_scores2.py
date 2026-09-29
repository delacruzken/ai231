import sys
sys.path.insert(0, "/Users/hdc/sandbox/ai231/me2")
import torch
from grammar_asr.asr_model_res import ResCTCEncoder
from grammar_asr.asr_model import greedy_decode
from grammar_asr.features import load_wav, log_mel, normalize_feat, load_manifest
from grammar_asr.grammar import build_grammar, NUM_CLASSES, idx_to_char
from grammar_asr.decode_grammar import score_all_words, forced_align_score
from grammar_asr.grammar import char_to_idx

device = "mps"
model = ResCTCEncoder(hidden=192, num_blocks=4).to(device)
model.load_state_dict(torch.load("grammar_asr/runs/res30/best.pt", map_location=device))
model.eval()
trie = build_grammar()

samples = [s for s in load_manifest(split="test", condition="clean") if s.intent == "ALARM"][:2]
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
    # score a few specific words
    for w in ["time", "stop", "call", "alarm six am", "wake me up at six am",
              "set an alarm for six am", "play music", "turn on the lights"]:
        chars = [char_to_idx[c] for c in w if c in char_to_idx]
        s_fa, _ = forced_align_score(log_probs, chars, NUM_CLASSES-1, length_norm=True)
        s_raw, _ = forced_align_score(log_probs, chars, NUM_CLASSES-1, length_norm=False)
        print(f"    {w:28s} len_norm={s_fa:7.3f}  raw={s_raw:8.2f}  L={len(chars)}")
    print("  --- top-8 by length-norm score ---")
    res = score_all_words(log_probs, trie, length_norm=True)[:8]
    for r in res:
        print(f"    {r.score:7.3f}  {r.intent:16s} {r.transcript!r}")
    print()
