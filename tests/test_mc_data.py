import random

from sindhi.mc_data import to_row, valid


def test_valid_and_shuffle_keep_the_right_answer():
    q = {"question": "ڪير؟", "options": ["الف", "ب", "ت", "ث"], "answer": 2}
    assert valid(q)
    assert not valid({**q, "options": ["الف", "الف", "ت", "ث"]})
    assert not valid({**q, "answer": 4})
    letters = set()
    for s in range(20):
        row = to_row("متن", q, random.Random(s))
        ans = row["messages"][1]["content"]
        assert ans.endswith(") ت")
        letters.add(ans[0])
        assert f"{ans[0]}) ت" in row["messages"][0]["content"]
    assert len(letters) > 1
