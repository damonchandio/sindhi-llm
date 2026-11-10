"""Download the phone recordings (recorder/index.html) into data/voice/: wavs/<id>.wav (the latest take of each
sentence) and metadata.csv ("id|text", the LJSpeech layout TTS trainers read). Re-running fetches only new takes.
    python -m sindhi.voice     # needs data/voice/secret.txt (the part after # in the recorder link)
"""
from __future__ import annotations

import argparse
import json
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "voice"
STORE = "https://ibrmcqqxowzhpjgfmmsx.supabase.co/storage/v1"
KEY = "sb_publishable_AgEWnlG6Kk9buromI6sPEA_MOS65btA"   # publishable: safe in public code, the secret is what guards
BUCKET, SINCE = "sindhi-voice", 1793311131412             # same cut-off as the recorder: set-up test clips ignored
NAME = re.compile(r"^(\d{4})-(\d+)(?:-(\d+)ms)?\.(wav|json)$")


def call(secret: str, path: str, body: dict | None = None) -> bytes:
    req = urllib.request.Request(f"{STORE}/{path}", data=json.dumps(body).encode() if body else None,
                                 headers={"apikey": KEY, "Authorization": f"Bearer {KEY}", "x-voice-key": secret,
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def latest(names: list[str], since: int) -> dict[str, tuple[str, int]]:
    """id -> (file name, milliseconds; 0 = skipped), keeping the newest take of each sentence."""
    out: dict[str, tuple[int, str, int]] = {}
    for n in names:
        m = NAME.match(n)
        if m and int(m[2]) >= since and (m[1] not in out or int(m[2]) > out[m[1]][0]):
            out[m[1]] = (int(m[2]), n, int(m[3] or 0) if m[4] == "wav" else 0)
    return {k: (n, ms) for k, (_, n, ms) in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=SINCE)
    a = ap.parse_args()
    secret = (OUT / "secret.txt").read_text().strip().lstrip("#").split("#")[-1]
    names, offset = [], 0
    while True:
        page = json.loads(call(secret, f"object/list/{BUCKET}", {"prefix": secret + "/", "limit": 1000, "offset": offset}))
        names += [p["name"] for p in page]
        if len(page) < 1000:
            break
        offset += 1000
    takes = latest(names, a.since)
    text = {i: t for i, _, t in json.loads((ROOT / "recorder" / "script.json").read_text(encoding="utf-8"))}
    (OUT / "wavs").mkdir(parents=True, exist_ok=True)
    have_file = OUT / "takes.json"
    have = json.loads(have_file.read_text()) if have_file.exists() else {}
    new = 0
    for i, (n, ms) in sorted(takes.items()):
        if ms and have.get(i) != n:
            (OUT / "wavs" / f"{i}.wav").write_bytes(call(secret, f"object/authenticated/{BUCKET}/{secret}/{n}"))
            have[i] = n
            new += 1
    have_file.write_text(json.dumps(have, indent=0))
    clips = sorted(i for i, (_, ms) in takes.items() if ms)
    (OUT / "metadata.csv").write_text("".join(f"{i}|{text[i]}\n" for i in clips), encoding="utf-8")
    minutes = sum(ms for _, ms in takes.values()) / 60000
    print(f"{len(clips)} recordings ({minutes:.1f} min), {len(takes) - len(clips)} skipped, {new} new downloaded")


if __name__ == "__main__":
    main()
