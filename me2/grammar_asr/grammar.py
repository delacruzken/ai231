"""
Grammar trie for ME2 spoken-command recognition.

Builds a character-level trie from the 19 intents and their phrase templates.
Used by the constrained decoder to walk predicted characters and emit only
valid commands (PocketSphinx-style grammar-constrained decoding).

Character alphabet: lowercase a-z, space, blank (CTC). Numbers are spelled
out in the transcripts (e.g. "10 seconds") so no digit symbols are needed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "AI231_src" / "MEX2" / "OptionB"


# ---------------------------------------------------------------------------
# Character alphabet
# ---------------------------------------------------------------------------
BLANK = "<blk>"
SPACE = " "
CHARS = [chr(c) for c in range(ord("a"), ord("z") + 1)]  # a..z
ALPHABET = CHARS + [SPACE]            # 27 symbols
NUM_CLASSES = len(ALPHABET) + 1       # +1 for CTC blank = 28

char_to_idx: Dict[str, int] = {c: i for i, c in enumerate(ALPHABET)}
char_to_idx[BLANK] = NUM_CLASSES - 1
idx_to_char: List[str] = ALPHABET + [BLANK]


_DIGITS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
_TEENS = {
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen",
}
_TENS = {
    "20": "twenty", "30": "thirty", "40": "forty", "50": "fifty",
    "60": "sixty", "70": "seventy", "80": "eighty", "90": "ninety",
}


def _num_to_words(n: int) -> str:
    """Spell a non-negative integer in English (good enough for 0-999)."""
    if n == 0:
        return "zero"
    parts = []
    if n >= 100:
        parts.append(_DIGITS[str(n // 100)] + " hundred")
        n %= 100
    if n >= 20:
        t, o = divmod(n, 10)
        word = _TENS[str(t * 10)]
        if o:
            word += " " + _DIGITS[str(o)]
        parts.append(word)
    elif 10 <= n < 20:
        parts.append(_TEENS[str(n)])
    elif 0 < n < 10:
        parts.append(_DIGITS[str(n)])
    return " ".join(p for p in parts if p)


def normalize(text: str) -> str:
    """Lowercase, spell out digits, strip punctuation, collapse whitespace.

    Digits are spelled the way the TTS pronounced them, so the grammar and the
    ASR target share one representation (e.g. '10 seconds' -> 'ten seconds').
    """
    t = text.lower()
    out = []
    i = 0
    while i < len(t):
        ch = t[i]
        if ch.isdigit():
            j = i
            while j < len(t) and t[j].isdigit():
                j += 1
            out.append(" " + _num_to_words(int(t[i:j])))
            i = j
        elif ch.isalpha():
            out.append(ch)
            i += 1
        elif ch == " ":
            out.append(" ")
            i += 1
        else:
            i += 1  # drop punctuation
    # collapse multiple spaces
    collapsed = []
    prev_space = False
    for ch in out:
        if ch == " ":
            if not prev_space and collapsed:
                collapsed.append(" ")
            prev_space = True
        else:
            collapsed.append(ch)
            prev_space = False
    return "".join(collapsed).strip()


# ---------------------------------------------------------------------------
# Trie
# ---------------------------------------------------------------------------
@dataclass
class TrieNode:
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    is_terminal: bool = False
    intents: Set[str] = field(default_factory=set)
    transcripts: Set[str] = field(default_factory=set)


class GrammarTrie:
    def __init__(self) -> None:
        self.root = TrieNode()
        self.num_words = 0

    def add(self, transcript: str, intent: str) -> None:
        norm = normalize(transcript)
        if not norm:
            return
        node = self.root
        for ch in norm:
            node = node.children.setdefault(ch, TrieNode())
        node.is_terminal = True
        node.intents.add(intent)
        node.transcripts.add(norm)
        self.num_words += 1

    def add_many(self, pairs: List[Tuple[str, str]]) -> None:
        for t, i in pairs:
            self.add(t, i)

    # -- decoding helpers ----------------------------------------------------
    def next_chars(self, node: TrieNode) -> List[str]:
        return list(node.children.keys())

    def is_accept(self, node: TrieNode) -> bool:
        return node.is_terminal

    def intents_at(self, node: TrieNode) -> Set[str]:
        return node.intents

    # -- introspection -------------------------------------------------------
    def max_depth(self) -> int:
        def depth(n: TrieNode, d: int) -> int:
            if not n.children:
                return d
            return max(depth(c, d + 1) for c in n.children.values())
        return depth(self.root, 0)

    def num_nodes(self) -> int:
        def count(n: TrieNode) -> int:
            return 1 + sum(count(c) for c in n.children.values())
        return count(self.root)

    def all_transcripts(self) -> List[str]:
        out: List[str] = []

        def walk(n: TrieNode, prefix: str) -> None:
            if n.is_terminal:
                out.append(prefix)
            for ch, c in sorted(n.children.items()):
                walk(c, prefix + ch)

        walk(self.root, "")
        return out


# ---------------------------------------------------------------------------
# Build from dataset
# ---------------------------------------------------------------------------
def load_transcripts_from_manifest(manifest: Optional[Path] = None) -> List[Tuple[str, str]]:
    """Return [(transcript, intent)] unique pairs from manifest.csv."""
    manifest = manifest or (DATA_DIR / "manifest.csv")
    seen = set()
    pairs: List[Tuple[str, str]] = []
    with open(manifest, newline="", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(",")
        ti = header.index("transcript")
        ii = header.index("intent")
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) <= max(ti, ii):
                continue
            tr, it = parts[ti], parts[ii]
            key = (tr, it)
            if key not in seen:
                seen.add(key)
                pairs.append((tr, it))
    return pairs


GRAMMAR_JSON = Path(__file__).resolve().parent / "grammar_commands.json"


def _load_pairs(manifest: Optional[Path]) -> List[Tuple[str, str]]:
    """Prefer an explicit manifest; else the dataset manifest; else the
    bundled static grammar_commands.json (deploy path, no data dir needed)."""
    if manifest is not None:
        return load_transcripts_from_manifest(manifest)
    ds_manifest = DATA_DIR / "manifest.csv"
    if ds_manifest.exists():
        return load_transcripts_from_manifest(ds_manifest)
    if GRAMMAR_JSON.exists():
        data = json.loads(GRAMMAR_JSON.read_text(encoding="utf-8"))
        return [(d["transcript"], d["intent"]) for d in data]
    raise FileNotFoundError(
        f"No manifest at {ds_manifest} and no {GRAMMAR_JSON.name} beside grammar.py")


def build_grammar(manifest: Optional[Path] = None) -> GrammarTrie:
    trie = GrammarTrie()
    trie.add_many(_load_pairs(manifest))
    return trie


if __name__ == "__main__":
    g = build_grammar()
    print(f"Unique command words in grammar : {g.num_words}")
    print(f"Trie nodes                      : {g.num_nodes()}")
    print(f"Max command length (chars)      : {g.max_depth()}")
    print(f"Alphabet size (incl. blank)     : {NUM_CLASSES}")
    print("\nSample grammar entries:")
    for t in g.all_transcripts()[:15]:
        print(f"  {t!r}")
