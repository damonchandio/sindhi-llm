from types import SimpleNamespace

from sindhi.chat_train import encode


class CharTok:
    """One id per character, so masking is easy to check by eye."""
    def apply_chat_template(self, msgs, **kw):
        return "".join(f"<{m['role']}>{m['content']}" for m in msgs) + "<assistant>"

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(c) for c in text])


def test_loss_only_on_the_answer():
    e = encode(CharTok(), [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}], 200)
    prompt_len = len("<user>hi<assistant>")
    assert e["labels"][:prompt_len] == [-100] * prompt_len
    assert "".join(map(chr, e["labels"][prompt_len:])) == "yo<|im_end|>\n"


def test_prompt_too_long_is_dropped():
    assert encode(CharTok(), [{"role": "user", "content": "x" * 100}, {"role": "assistant", "content": "y"}], 64) is None
