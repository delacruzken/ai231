"""Character trie built from the canonical Option-B schema."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from ..schema import VARIATIONS, normalize_text, Variation

GRAMMAR_JSON = Path(__file__).resolve().parents[1] / "grammar_commands.json"


@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    is_terminal: bool = False
    intents: Set[str] = field(default_factory=set)
    transcripts: Set[str] = field(default_factory=set)
    variation_ids: Set[int] = field(default_factory=set)


class GrammarTrie:
    def __init__(self) -> None:
        self.root = TrieNode()
        self.num_words = 0
        self.variations: List[Variation] = []

    def add(self, variation: Variation) -> None:
        norm = variation.normalized
        if not norm:
            return
        node = self.root
        for ch in norm:
            node = node.children.setdefault(ch, TrieNode())
        node.is_terminal = True
        node.intents.add(variation.intent)
        node.transcripts.add(norm)
        node.variation_ids.add(variation.variation_id)
        self.num_words += 1
        self.variations.append(variation)

    def all_transcripts(self) -> List[str]:
        out: List[str] = []

        def walk(n: TrieNode, prefix: str) -> None:
            if n.is_terminal:
                out.append(prefix)
            for ch, c in sorted(n.children.items()):
                walk(c, prefix + ch)

        walk(self.root, "")
        return out

    def variation_for_norm(self, norm: str) -> Optional[Variation]:
        node = self.root
        for ch in norm:
            if ch not in node.children:
                return None
            node = node.children[ch]
        if not node.is_terminal:
            return None
        vid = next(iter(node.variation_ids))
        for v in self.variations:
            if v.variation_id == vid:
                return v
        return None


def build_grammar(variations=None) -> GrammarTrie:
    trie = GrammarTrie()
    for v in (variations or VARIATIONS):
        trie.add(v)
    return trie


def export_grammar_json(path: Optional[Path] = None) -> Path:
    path = path or GRAMMAR_JSON
    data = [
        {
            "transcript": v.phrase,
            "intent": v.intent,
            "slot": v.slot_value,
            "variation_id": v.variation_id,
            "normalized": v.normalized,
        }
        for v in VARIATIONS
    ]
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path
