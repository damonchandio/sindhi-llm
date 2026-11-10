from pathlib import Path

from sindhi.corpus import sindhi_enough
from sindhi.train_big import newest


def test_sindhi_vs_urdu():
    assert sindhi_enough(["سنڌي ٻولي ۾ ڪيترائي اکر آهن جيڪي اردوءَ ۾ نه آهن"])
    assert not sindhi_enough(["اردو ایک خوبصورت زبان ہے جو پاکستان میں بولی جاتی ہے"])
    assert not sindhi_enough(["English only"])


def test_newest_checkpoint_is_by_step_not_name(tmp_path: Path):
    for s in (25, 100, 75):
        (tmp_path / f"checkpoint-{s}").mkdir()
        (tmp_path / f"checkpoint-{s}" / "trainer_state.json").write_text("{}")
    (tmp_path / "checkpoint-125").mkdir()   # half-written save from a power cut: skipped
    assert newest(tmp_path).name == "checkpoint-100"
    assert newest(tmp_path / "missing") is None
