"""A blind A/B page for a Sindhi speaker: each prompt with the base and trained completions in random order.
Tap the better one; the page keeps score and shows the result. No typing.
    python -m sindhi.compare                       -> results/compare.html (base.json vs lora.json)
    python -m sindhi.compare base-4b lora-4b       -> results/compare-lora-4b.html
"""
import html
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
A, B = sys.argv[1:3] if len(sys.argv) > 2 else ("base", "lora")
base = json.loads((ROOT / "results" / f"{A}.json").read_text(encoding="utf-8"))["samples"]
lora = json.loads((ROOT / "results" / f"{B}.json").read_text(encoding="utf-8"))["samples"]
rng = random.Random(3)
items = []
for p in base:
    pair = [("base", base[p]), ("lora", lora[p])]
    rng.shuffle(pair)
    items.append({"prompt": p, "a": pair[0], "b": pair[1]})
page = """<!doctype html><html lang="sd" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sindhi A/B</title><style>body{font-family:'Noto Naskh Arabic','Segoe UI',sans-serif;background:#0b1724;color:#fff;margin:0;padding:20px;font-size:22px}
.q{margin:0 0 10px;color:#2aa6f5;font-size:28px}.c{display:block;width:100%;text-align:right;background:#132638;border:2px solid #2b4a66;color:#fff;border-radius:14px;padding:18px;margin:10px 0;font:inherit;line-height:1.7}
.c:active{border-color:#2aa6f5}#done{font-size:30px;text-align:center;margin-top:60px}small{color:#9fb3c6;font-size:16px;display:block;text-align:left;direction:ltr}</style>
<body><div id="app"></div><script>
const items=ITEMS;let i=0,score={base:0,lora:0,tie:0};const app=document.getElementById('app');
function show(){if(i>=items.length){const t=score.base+score.lora+score.tie;app.innerHTML=`<div id=done>Done.<br>trained ${score.lora} · base ${score.base} · same ${score.tie}<br><small>${t} prompts. Copy this line into results/human.txt</small></div>`;return}
const it=items[i];app.innerHTML=`<small>${i+1} / ${items.length}</small><p class=q>${it.prompt}…</p>
<button class=c onclick="pick('a')">${it.a[1]}</button><button class=c onclick="pick('b')">${it.b[1]}</button><button class=c onclick="pick('tie')" style="text-align:center;color:#9fb3c6">ٻئي ساڳيا / both as bad</button>`}
function pick(k){score[k==='tie'?'tie':items[i][k][0]]++;i++;show()}show();
</script></body></html>"""
out = ROOT / "results" / ("compare.html" if B == "lora" else f"compare-{B}.html")
out.write_text(page.replace("ITEMS", json.dumps(items, ensure_ascii=False)), encoding="utf-8")
print("open", out)
