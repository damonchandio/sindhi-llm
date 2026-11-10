from sindhi.voice import latest


def test_newest_take_wins_and_old_test_clips_are_ignored():
    t = latest(["0001-100-2000ms.wav", "0001-200.json", "0002-300-1500ms.wav", "0002-400-1800ms.wav",
                "probe.json", "0003-50-900ms.wav"], since=100)
    assert t == {"0001": ("0001-200.json", 0), "0002": ("0002-400-1800ms.wav", 1800)}
