import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sindhi.data import clean  # noqa: E402


def test_clean_keeps_sindhi_prose_only():
    text = ("سنڌي ٻولي هڪ قديم ٻولي آهي جيڪا سنڌ ۾ ڳالهائي وڃي ٿي ۽ ان جو ادب تمام شاهوڪار آهي.\n"
            "short\n"
            "This is an English paragraph that is long enough but has no Sindhi letters in it at all.\n")
    out = clean(text)
    assert len(out) == 1 and out[0].startswith("سنڌي")
