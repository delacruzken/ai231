"""Training losses for CTC / intent-CRNN / hybrid models."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..schema import (
    INTENT_TO_ID, OUT_OF_SCOPE, SLOT_INTENTS, SLOT_VALUE_TO_ID, NUM_INTENT_OOS,
)
from ..text import encode_transcript, NUM_CLASSES
from ..models.ctc_low import post_len_low


CONFUSABLE_PAIRS = [
    ("LIGHT_ON", "LIGHT_OFF"),
    ("VOLUME_UP", "VOLUME_DOWN"),
    ("PAUSE", "STOP"),
    ("PAUSE", "NEXT"),
    ("STOP", "NEXT"),
]


def encode_targets(transcripts: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = [encode_transcript(t) for t in transcripts]
    lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    max_l = int(lengths.max()) if len(encoded) else 1
    targets = torch.zeros(len(encoded), max_l, dtype=torch.long)
    for i, e in enumerate(encoded):
        if e:
            targets[i, :len(e)] = torch.tensor(e, dtype=torch.long)
    return targets, lengths


def ctc_loss_fn(logits, targets, input_lengths, target_lengths, blank=None):
    blank = NUM_CLASSES - 1 if blank is None else blank
    # logits: (B, T, C) -> (T, B, C)
    # CTC's long reductions are numerically unstable in float16 under AMP.
    log_probs = F.log_softmax(logits.float(), dim=-1).transpose(0, 1)
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, zero_infinity=True)


def intent_targets(intents: Sequence[str], oos_flags: Optional[torch.Tensor] = None):
    ids = []
    for i, name in enumerate(intents):
        if oos_flags is not None and bool(oos_flags[i]):
            ids.append(INTENT_TO_ID[OUT_OF_SCOPE])
        else:
            ids.append(INTENT_TO_ID.get(name, INTENT_TO_ID[OUT_OF_SCOPE]))
    return torch.tensor(ids, dtype=torch.long)


def slot_targets(intents: Sequence[str], slots: Sequence[str]):
    """Return dict intent -> (index_tensor, mask) for slotted samples."""
    out = {}
    for intent in SLOT_INTENTS:
        idxs = []
        mask = []
        vocab = SLOT_VALUE_TO_ID[intent]
        for name, slot in zip(intents, slots):
            if name == intent and slot in vocab:
                idxs.append(vocab[slot])
                mask.append(True)
            else:
                idxs.append(0)
                mask.append(False)
        out[intent] = (
            torch.tensor(idxs, dtype=torch.long),
            torch.tensor(mask, dtype=torch.bool),
        )
    return out


def confusable_pair_penalty(intent_logits: torch.Tensor,
                            intent_ids: torch.Tensor,
                            weight: float = 0.1) -> torch.Tensor:
    """Encourage margin between confusable intent pairs."""
    if weight <= 0:
        return intent_logits.new_zeros(())
    probs = F.log_softmax(intent_logits, dim=-1)
    loss = intent_logits.new_zeros(())
    n = 0
    for a, b in CONFUSABLE_PAIRS:
        ia, ib = INTENT_TO_ID[a], INTENT_TO_ID[b]
        # for samples of class a, penalize high logit on b and vice versa
        for pos, neg in ((ia, ib), (ib, ia)):
            m = intent_ids == pos
            if not bool(m.any()):
                continue
            # hinge on logp(pos) - logp(neg)
            margin = probs[m, pos] - probs[m, neg]
            loss = loss + F.relu(0.5 - margin).mean()
            n += 1
    if n == 0:
        return intent_logits.new_zeros(())
    return weight * loss / n


def multitask_loss(outputs: dict, batch: dict, *,
                   ctc_weight: float = 1.0,
                   intent_weight: float = 1.0,
                   slot_weight: float = 1.0,
                   confusable_weight: float = 0.0,
                   label_smoothing: float = 0.0) -> Tuple[torch.Tensor, dict]:
    device = None
    for v in outputs.values():
        if torch.is_tensor(v):
            device = v.device
            break
        if isinstance(v, dict):
            for vv in v.values():
                if torch.is_tensor(vv):
                    device = vv.device
                    break
    if device is None:
        raise ValueError("outputs contained no tensors")

    parts = {}
    total = torch.zeros((), device=device)

    if "ctc_logits" in outputs and ctc_weight > 0:
        logits = outputs["ctc_logits"]
        T_raw = batch["T"].to(device)
        T_post = torch.tensor([post_len_low(int(t)) for t in T_raw.tolist()],
                              dtype=torch.long, device=device)
        # Skip OOS for CTC if transcript empty / non-command
        keep = []
        transcripts = []
        for i, (tr, oos) in enumerate(zip(batch["transcripts"], batch["out_of_scope"])):
            if (not bool(oos)) and tr and encode_transcript(tr):
                keep.append(i)
                transcripts.append(tr)
        if keep:
            idx = torch.tensor(keep, device=device)
            targets, tgt_lens = encode_targets(transcripts)
            targets = targets.to(device)
            tgt_lens = tgt_lens.to(device)
            T_post_k = T_post[idx]
            valid = tgt_lens <= T_post_k
            if bool(valid.any()):
                idx2 = valid.nonzero(as_tuple=True)[0]
                loss_c = ctc_loss_fn(
                    logits[idx][idx2], targets[idx2], T_post_k[idx2], tgt_lens[idx2]
                )
                parts["ctc"] = float(loss_c.detach())
                total = total + ctc_weight * loss_c

    if "intent_logits" in outputs and intent_weight > 0:
        y = intent_targets(batch["intent"], batch["out_of_scope"]).to(device)
        loss_i = F.cross_entropy(outputs["intent_logits"], y,
                                 label_smoothing=label_smoothing)
        parts["intent"] = float(loss_i.detach())
        total = total + intent_weight * loss_i
        if confusable_weight > 0:
            pen = confusable_pair_penalty(outputs["intent_logits"], y,
                                          weight=confusable_weight)
            parts["confusable"] = float(pen.detach())
            total = total + pen

    if "slot_logits" in outputs and slot_weight > 0:
        slot_t = slot_targets(batch["intent"], batch["slot"])
        slot_losses = []
        for intent, (tgt, mask) in slot_t.items():
            if not bool(mask.any()):
                continue
            logits = outputs["slot_logits"][intent]
            tgt = tgt.to(device)
            mask = mask.to(device)
            loss_s = F.cross_entropy(logits[mask], tgt[mask],
                                     label_smoothing=label_smoothing)
            slot_losses.append(loss_s)
        if slot_losses:
            loss_slot = torch.stack(slot_losses).mean()
            parts["slot"] = float(loss_slot.detach())
            total = total + slot_weight * loss_slot

    if "wake_logits" in outputs:
        # batch must provide wake labels 0/1 in batch["wake"]
        y = batch["wake"].to(device)
        loss_w = F.cross_entropy(outputs["wake_logits"], y)
        parts["wake"] = float(loss_w.detach())
        total = total + loss_w

    parts["total"] = float(total.detach())
    return total, parts
