from grammar_asr.data.dataset import (
    Sample, make_speaker_val_split, check_speaker_leakage, dataset_fingerprint,
)


def _s(split, speaker, file, oos=False):
    return Sample(
        path=None, audio=None, split=split,
        intent="OUT_OF_SCOPE" if oos else "TIME",
        slot_value="", variation="" if oos else "Time", variation_id=-1 if oos else 0,
        transcript="time", out_of_scope=oos, speaker_id=speaker, source="unit",
        is_synthetic=False, accent_group="Unknown", duration_s=1.0, file=file,
    )


def test_speaker_val_split_disjoint():
    train = [_s("train", f"spk{i//2}", f"f{i}.wav") for i in range(20)]
    tr, va = make_speaker_val_split(train, val_fraction=0.2, seed=0)
    assert check_speaker_leakage({"train": tr, "val": va})["ok"]
    assert len(tr) + len(va) == len(train)


def test_fingerprint_changes_with_files():
    a = {"train": [_s("train", "a", "1.wav")], "test": [_s("test", "b", "2.wav")]}
    b = {"train": [_s("train", "a", "1.wav")], "test": [_s("test", "b", "3.wav")]}
    fa = dataset_fingerprint(a)
    fb = dataset_fingerprint(b)
    assert fa["sha256_file_list"] != fb["sha256_file_list"]
    assert fa["speaker_leakage"]["ok"]
