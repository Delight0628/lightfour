"""
103 音色人工标注工具（本地，零外部依赖，不外传音频）。

用途：
  408 条种子的 `human_notation` 目前 408 行全空，是 §10.1 #5 的硬阻塞。
  本工具把「切片 + 参考音对照 + 下拉选择 + 导出」做成可操作闭环，
  让人工标注能在通道断开的本地环境里先跑起来。

启动：
  python scripts/annotate_ui.py --port 8777
  浏览器打开 http://127.0.0.1:8777

只读：
  - data/sound_seed_annotation.csv（待标注队列）
  - data/beatbox_taxonomy.json（103 音色下拉）
  - data/beatbox_seed_crosswalk.csv（候选记法 → canonical）
  - 音频文件（本地可读的切片 + data/beatbox_seeds/audio 参考音）

写：
  - data/sound_seed_annotation_human.csv（人工填写结果，不覆盖原 CSV）

说明：音频只在本地 127.0.0.1 环回提供，不出网。
"""
import argparse
import csv
import json
import threading
import webbrowser
from collections import defaultdict
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
ANNOT_CSV = DATA / "sound_seed_annotation.csv"
OUT_CSV = DATA / "sound_seed_annotation_human.csv"
TAXONOMY = DATA / "beatbox_taxonomy.json"
CROSSWALK = DATA / "beatbox_seed_crosswalk.csv"
SEED_AUDIO = DATA / "beatbox_seeds" / "audio"


def load_queue():
    rows = list(csv.DictReader(open(ANNOT_CSV, encoding="utf-8-sig")))
    tax = json.loads(TAXONOMY.read_text(encoding="utf-8"))
    cw = {c["seedlib_notation"]: c for c in csv.DictReader(open(CROSSWALK, encoding="utf-8"))}

    # 参考音：canonical sound_id -> 本地 mp3
    ref = {}
    sd = json.loads((DATA / "beatbox_seeds" / "sounds.json").read_text(encoding="utf-8"))
    for f in sorted(SEED_AUDIO.glob("*.mp3")):
        head = f.stem.split("_")[0]
        if not head.isdigit():
            continue
        idx = int(head) - 1
        if idx >= len(sd):
            continue
        c = cw.get(sd[idx]["notation"])
        if c and c["canonical_sound_id"]:
            ref[c["canonical_sound_id"]] = {
                "path": str(f.relative_to(BASE)).replace("\\", "/"),
                "name": c["canonical_en"],
                "notation": c["canonical_notation"],
            }

    # 本地可读的切片音频：sound_seed_pack 命名为 "{Family}_{id}.wav"，
    # 候选全集目录命名为 "{id}.wav"（408 条，从远端持久盘拉回）
    local = {}
    pack = DATA / "bbox_raw" / "sound_seed_pack"
    if pack.exists():
        for f in pack.glob("*.wav"):
            stem = f.stem
            local[stem] = str(f.relative_to(BASE)).replace("\\", "/")
            # 同时按去前缀的 id 建索引，兼容 "Family_id" 与 "id" 两种命名
            local.setdefault(stem.split("_", 1)[-1], str(f.relative_to(BASE)).replace("\\", "/"))
    for cand_root in (DATA / "bbox_raw" / "seed_candidates" / "cand408",
                      DATA / "bbox_raw" / "seed_candidates"):
        if not cand_root.exists():
            continue
        for f in cand_root.glob("*.wav"):
            local.setdefault(f.stem, str(f.relative_to(BASE)).replace("\\", "/"))

    items = []
    for r in rows:
        sid = r.get("sound_id", "")
        stem = Path(r.get("wav", "")).stem
        items.append({
            "id": r.get("id"),
            "source": r.get("source"),
            "split": r.get("split"),
            "wav": r.get("wav"),
            "local_wav": local.get(stem, ""),
            "candidate_notation": r.get("candidate_notation"),
            "candidate_family_l2": r.get("family"),
            "canonical_notation": r.get("canonical_notation"),
            "canonical_en": r.get("canonical_en"),
            "canonical_family": r.get("canonical_family"),
            "sound_id": sid,
            "nearest_seed": r.get("nearest_seed"),
            "human_notation": r.get("human_notation", ""),
            "note": r.get("note", ""),
            "annotate_status": r.get("annotate_status", "pending"),
            "annotator": r.get("annotator", ""),
            "ref": ref.get(sid),
        })
    return items, tax, ref


def build_plan(items, min_per_family=5, target_families=15):
    """标注优先级工作单。

    PI 门槛（§10.1 #5）：human_notation 覆盖 canonical family >= 15/19，
    且每 family n >= 5。因此工作单按「还差多少条才能让该 family 达标」排序：
      - 优先补 family 覆盖数（每 family 至少 1 条 -> 覆盖 family 数）
      - 同一 family 内优先补到 n>=5
      - 同一 family 内优先选不同 sound_id（覆盖更多音色）

    返回每个 family 的目标条数与条目 id 列表（不含音频内容）。
    """
    by_fam = defaultdict(list)
    for it in items:
        by_fam[it["canonical_family"] or "unknown"].append(it)

    # 已确认的条数（按当前 human_notation + status）
    def is_done(it):
        return bool(it.get("human_notation")) and it.get("annotate_status") == "confirmed"

    plan = {}
    for fam, lst in by_fam.items():
        done = [it for it in lst if is_done(it)]
        need = max(0, min_per_family - len(done))
        # family 内按 sound_id 多样性排序：先取每个 sound_id 一条，再补余量
        seen_sid = set()
        first, rest = [], []
        for it in sorted(lst, key=lambda x: (x.get("split") != "train", x["id"])):
            sid = it.get("sound_id")
            if sid not in seen_sid:
                seen_sid.add(sid)
                first.append(it)
            else:
                rest.append(it)
        ordered = first + rest
        target = ordered[:max(need, min(len(ordered), min_per_family))]
        plan[fam] = {
            "total": len(lst),
            "confirmed": len(done),
            "need": need,
            "target_ids": [it["id"] for it in target],
            "sound_id_slots": len({it.get("sound_id") for it in ordered}),
        }
    return plan


def export_worklist(items, out_csv="data/annotation_worklist.csv",
                    min_per_family=5, target_families=15):
    """导出标注工作单：达到 PI 门槛所需的最小标注集合。

    门槛（§10.1 #5）：human_notation 覆盖 canonical family >= 15/19，
    且每 family n >= 5。工作单按此最小化，并附参考音路径，
    使标注者（或能听音的代理）能一条一条直接判定。

    列：
      order / id / sound_family / sound_id / candidate_notation / canonical_notation
      canonical_en / slice_wav / ref_wav / ref_notation / split / source
      human_notation(待填) / annotate_status(待填) / annotator(待填) / note(待填)
    """
    plan = build_plan(items, min_per_family=min_per_family,
                      target_families=target_families)
    by_id = {it["id"]: it for it in items}
    rows = []
    order = 0
    # 先做 family 覆盖：每个 family 先取 1 条（优先 sound_id 多样）
    for fam in sorted(plan):
        p = plan[fam]
        if p["need"] <= 0:
            continue
        ids = p["target_ids"]
        if not ids:
            continue
        # 该 family 第一条（覆盖用）
        chosen = [ids[0]]
        # 再补到 min_per_family
        chosen += ids[1:max(1, min_per_family - p["confirmed"])]
        for sid in chosen:
            it = by_id[sid]
            order += 1
            rows.append({
                "order": order,
                "id": sid,
                "sound_family": it["canonical_family"],
                "sound_id": it.get("sound_id"),
                "candidate_notation": it.get("candidate_notation"),
                "canonical_notation": it.get("canonical_notation"),
                "canonical_en": it.get("canonical_en"),
                "slice_wav": it.get("local_wav"),
                "ref_wav": (it.get("ref") or {}).get("path", ""),
                "ref_notation": (it.get("ref") or {}).get("notation", ""),
                "split": it.get("split"),
                "source": it.get("source"),
                "human_notation": "",
                "annotate_status": "",
                "annotator": "",
                "note": "",
            })
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    fams = {r["sound_family"] for r in rows}
    return out, len(rows), len(fams)


def save_annotations(payload):
    """payload: [{id, human_notation, note, annotate_status, annotator}]"""
    rows = list(csv.DictReader(open(ANNOT_CSV, encoding="utf-8-sig")))
    by_id = {p["id"]: p for p in payload}
    fields = list(rows[0].keys())
    for extra in ("annotate_status", "annotator"):
        if extra not in fields:
            fields.append(extra)
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            p = by_id.get(r.get("id"))
            if p:
                r["human_notation"] = p.get("human_notation", "")
                r["note"] = p.get("note", "")
                r["annotate_status"] = p.get("annotate_status", "")
                r["annotator"] = p.get("annotator", "")
            w.writerow(r)
    return OUT_CSV


PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>Lightfour 103 音色人工标注</title>
<style>
 body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#111;color:#eee}
 header{padding:10px 16px;background:#1c1c1c;position:sticky;top:0;z-index:5}
 h1{font-size:16px;margin:0 0 6px}
 .bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:13px}
 button{background:#2d6cdf;color:#fff;border:0;border-radius:4px;padding:6px 12px;cursor:pointer}
 button.ghost{background:#444}
 select,input{background:#222;color:#eee;border:1px solid #444;border-radius:4px;padding:5px}
 #list{max-height:180px;overflow:auto;margin-top:8px;font-size:12px}
 #list div{padding:3px 6px;cursor:pointer;border-radius:3px}
 #list div:hover{background:#333}
 #list div.done{color:#7fd18a}
 #list div.cur{background:#2d6cdf}
 .main{display:flex;gap:16px;padding:16px;align-items:flex-start;flex-wrap:wrap}
 .panel{background:#1a1a1a;border:1px solid #333;border-radius:6px;padding:12px}
 audio{width:340px;height:32px;display:block;margin:6px 0}
 table{border-collapse:collapse;font-size:13px}
 td,th{padding:4px 10px;border-bottom:1px solid #2a2a2a;text-align:left}
 .hint{font-size:12px;color:#999;max-width:420px;line-height:1.6}
 .ok{color:#7fd18a}.bad{color:#ff8080}
</style></head><body>
<header>
 <h1>Lightfour 103 音色人工标注</h1>
 <div class="bar">
  <span id="prog"></span>
  <button onclick="save()">保存全部</button>
  <button class="ghost" onclick="exportCsv()">导出 CSV</button>
  <label>筛选 <select id="filt" onchange="render()">
    <option value="all">全部</option><option value="pending">未标注</option>
    <option value="done">已标注</option><option value="work">工作单(优先)</option></select></label>
  <label>family <select id="fam" onchange="render()"></select></label>
  <label><input id="onlyplan" type="checkbox" checked onchange="render()"> 仅工作单目标</label>
 </div>
 <div id="list"></div>
</header>
<div class="main">
 <div class="panel">
  <table id="meta"></table>
  <div id="players"></div>
  <div style="margin-top:8px;font-size:12px">
   <b>快捷键</b>: <kbd>空格</kbd>=切片/参考音切换 · <kbd>1..9</kbd>=选下拉第n项 ·
   <kbd>Enter</kbd>=确认并跳下一条 · <kbd>U</kbd>=标 unknown · <kbd>←/→</kbd>=上/下一条
  </div>
 </div>
 <div class="panel">
  <b>标注</b>
  <div style="margin-top:8px">
   <select id="notsel" onchange="onPick()"></select>
   <input id="notfree" placeholder="或手输记法" style="width:110px">
  </div>
  <div style="margin-top:8px">
   <select id="status" onchange="mark()">
     <option value="pending">未确认</option>
     <option value="confirmed">已确认</option>
     <option value="rejected">否决(非该音色)</option>
     <option value="unknown">unknown</option>
   </select>
   <input id="annotator" placeholder="标注人" style="width:90px" onchange="mark()">
  </div>
  <input id="note" placeholder="备注" style="width:100%;margin-top:8px" onchange="mark()">
  <div class="hint" style="margin-top:10px">
   <b>规则</b><br>
   1. 先听「切片」，再听「参考音」对照判断。<br>
   2. 下拉是 canonical 103 音色；<b>不确定就选 unknown</b>，不要硬凑。<br>
   3. 该 family 至少确认 1 条，human_notation 覆盖 ≥15 family 前，
      任何识别准确率不得写入 §0。<br>
   4. 记法只填 canonical 记法（不带种子库的 <code>&lt;</code> 前缀）。
  </div>
 </div>
</div>
<script>
let ITEMS=[],TAX=[],REF={},CUR=-1,DIRTY=false,PLAN={},PLAYER=0;
async function init(){
 const q=new URLSearchParams(location.search);
 ITEMS=await (await fetch('/api/queue')).json();
 TAX=await (await fetch('/api/taxonomy')).json();
 REF=await (await fetch('/api/ref')).json();
 PLAN=await (await fetch('/api/plan')).json();
 const fams=[...new Set(ITEMS.map(i=>i.canonical_family))].sort();
 document.getElementById('fam').innerHTML='<option value="">全部</option>'+
   fams.map(f=>`<option>${f}</option>`).join('');
 document.getElementById('notsel').innerHTML='<option value="">— 选择 canonical 记法 —</option>'+
   TAX.map(t=>`<option value="${t.notation}" data-en="${t.en}" data-cn="${t.cn}">${t.notation} · ${t.cn} · ${t.en}</option>`).join('');
 render();
}
function planSet(){
 const s=new Set();
 for(const k in PLAN) for(const id of PLAN[k].target_ids) s.add(id);
 return s;
}
function filtered(){
 const f=document.getElementById('filt').value;
 const fam=document.getElementById('fam').value;
 const onlyplan=document.getElementById('onlyplan').checked;
 const ps=planSet();
 let l=ITEMS.map((x,i)=>({x,i}));
 if(fam) l=l.filter(o=>o.x.canonical_family===fam);
 if(onlyplan&&ps.size) l=l.filter(o=>ps.has(o.x.id));
 if(f==='pending') l=l.filter(o=>!o.x.human_notation);
 if(f==='done') l=l.filter(o=>o.x.human_notation);
 if(f==='work'){
   l.sort((a,b)=>{
     const pa=ps.has(a.x.id)?0:1, pb=ps.has(b.x.id)?0:1;
     if(pa!==pb) return pa-pb;
     const da=a.x.human_notation?1:0, db=b.x.human_notation?1:0;
     if(da!==db) return da-db;
     return a.x.id<b.x.id?-1:1;
   });
 }
 return l;
}
function render(){
 const l=filtered();
 const conf=ITEMS.filter(i=>i.human_notation&&i.annotate_status==='confirmed');
 const fams=new Set(conf.map(i=>i.canonical_family)).size;
 document.getElementById('prog').innerHTML=
   `共 ${ITEMS.length} 条 · 已确认 <b>${conf.length}</b> · family 覆盖 <b>${fams}/19</b> `+
   `· 门槛: family≥15 且每 family n≥5`;
 document.getElementById('list').innerHTML=l.map(o=>
   `<div class="${o.x.human_notation?'done':''} ${o.i===CUR?'cur':''}" onclick="pick(${o.i})">
     [${o.x.split}] ${o.x.id} · ${o.x.canonical_family} · ${o.x.canonical_en||'?'}${o.x.human_notation?' ✓ '+o.x.human_notation:''}</div>`).join('');
}
function pick(i){
 CUR=i; const x=ITEMS[i];
 document.getElementById('meta').innerHTML=`
  <tr><th>id</th><td>${x.id}</td></tr>
  <tr><th>source</th><td>${x.source} (${x.split})</td></tr>
  <tr><th>family</th><td>${x.canonical_family}</td></tr>
  <tr><th>候选</th><td>${x.candidate_notation||'-'} → ${x.canonical_notation||'-'} ${x.canonical_en||''}</td></tr>
  <tr><th>最近邻种子</th><td>${x.nearest_seed||'-'}</td></tr>
  <tr><th>切片源</th><td class="${x.local_wav?'ok':'bad'}">${x.local_wav||'本地不可读（需远端拉取）'}</td></tr>`;
 let ph='';
 if(x.local_wav) ph+=`<b>切片</b><audio id="a_slice" controls src="/${x.local_wav}"></audio>`;
 else ph+=`<b>切片</b><div class="bad">本地不可读</div>`;
 if(x.ref) ph+=`<b>参考音 ${x.ref.name} (${x.ref.notation})</b><audio id="a_ref" controls src="/${x.ref.path}"></audio>`;
 document.getElementById('players').innerHTML=ph;
 document.getElementById('notsel').value=x.human_notation||'';
 document.getElementById('notfree').value='';
 document.getElementById('status').value=x.annotate_status||'pending';
 document.getElementById('annotator').value=x.annotator||'';
 document.getElementById('note').value=x.note||'';
 render();
}
function onPick(){
 const s=document.getElementById('notsel'); const o=s.selectedOptions[0];
 if(o&&o.dataset.en) document.getElementById('notfree').value='';
}
function mark(){
 if(CUR<0)return; const x=ITEMS[CUR];
 x.human_notation=document.getElementById('notsel').value||document.getElementById('notfree').value.trim();
 x.annotate_status=document.getElementById('status').value;
 x.annotator=document.getElementById('annotator').value;
 x.note=document.getElementById('note').value;
 DIRTY=true; render();
}
function nextIdx(step){
 const l=filtered();
 if(!l.length) return -1;
 let pos=l.findIndex(o=>o.i===CUR);
 if(pos<0) pos=0; else pos=Math.min(l.length-1,Math.max(0,pos+step));
 return l[pos].i;
}
document.addEventListener('keydown',e=>{
 const tag=(e.target.tagName||'').toUpperCase();
 if(tag==='INPUT'||tag==='SELECT'||tag==='TEXTAREA') return;
 if(e.key===' '){ e.preventDefault();
   const a=document.getElementById(PLAYER===0?'a_slice':'a_ref');
   const b=document.getElementById(PLAYER===0?'a_ref':'a_slice');
   if(a){PLAYER=1;a.play();} else if(b){PLAYER=0;b.play();}
   return; }
 if(e.key>='1'&&e.key<='9'){ const s=document.getElementById('notsel');
   const idx=parseInt(e.key,10); if(s&&s.options[idx]){s.selectedIndex=idx;onPick();mark();} return; }
 if(e.key==='Enter'){ e.preventDefault();
   document.getElementById('status').value='confirmed'; mark();
   const ni=nextIdx(1); if(ni>=0) pick(ni); return; }
 if(e.key==='u'||e.key==='U'){ document.getElementById('status').value='unknown';
   document.getElementById('notsel').value=''; mark(); return; }
 if(e.key==='ArrowRight'){ const ni=nextIdx(1); if(ni>=0) pick(ni); return; }
 if(e.key==='ArrowLeft'){ const ni=nextIdx(-1); if(ni>=0) pick(ni); return; }
});
async function save(){
 const r=await fetch('/api/save',{method:'POST',
   headers:{'Content-Type':'application/json'},
   body:JSON.stringify(ITEMS.map(i=>({id:i.id,human_notation:i.human_notation,
     note:i.note,annotate_status:i.annotate_status,annotator:i.annotator})))});
 const t=await r.text(); alert(t); DIRTY=false;
}
async function exportCsv(){
 const r=await fetch('/api/save',{method:'POST',
   headers:{'Content-Type':'application/json'},
   body:JSON.stringify(ITEMS.map(i=>({id:i.id,human_notation:i.human_notation,
     note:i.note,annotate_status:i.annotate_status,annotator:i.annotator})))});
 window.location='/data/sound_seed_annotation_human.csv';
}
window.addEventListener('beforeunload',e=>{if(DIRTY){e.preventDefault();e.returnValue='';}});
init();
</script></body></html>"""


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(BASE), **kw)

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if p == "/api/queue":
            items, tax, ref = load_queue()
            self._json(items)
            return
        if p == "/api/taxonomy":
            tax = json.loads(TAXONOMY.read_text(encoding="utf-8"))
            self._json(tax["sounds"])
            return
        if p == "/api/ref":
            _i, _t, ref = load_queue()
            self._json(ref)
            return
        if p == "/api/plan":
            items, _tax, _ref = load_queue()
            self._json(build_plan(items))
            return
        super().do_GET()

    def do_POST(self):
        if urlparse(self.path).path != "/api/save":
            self.send_error(404)
            return
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n).decode("utf-8"))
        out = save_annotations(payload)
        done = sum(1 for p in payload if p.get("human_notation"))
        fams = {p.get("annotate_status") for p in payload}
        self._json({"ok": True, "path": str(out), "annotated": done,
                    "note": f"已保存 {done} 条到 {out}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8777)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--export-worklist", action="store_true",
                    help="只导出标注工作单 CSV 并退出（不启动服务）")
    ap.add_argument("--worklist-out", default="data/annotation_worklist.csv")
    args = ap.parse_args()

    items, tax, ref = load_queue()
    plan = build_plan(items)
    n_aud = sum(1 for i in items if i["local_wav"])
    done = sum(1 for i in items
               if i["human_notation"] and i["annotate_status"] == "confirmed")
    fam_done = {i["canonical_family"] for i in items
                if i["human_notation"] and i["annotate_status"] == "confirmed"}
    print(f"队列 {len(items)} 条；本地可听切片 {n_aud}；参考音 {len(ref)}")
    print(f"已确认 {done} 条；family 覆盖 {len(fam_done)}/19 "
          f"（门槛：family>=15 且每 family n>=5）")
    print(f"待补 family: " + ", ".join(
        f"{k}(+{v['need']})" for k, v in sorted(plan.items()) if v["need"] > 0))

    if args.export_worklist:
        out, n, fams = export_worklist(items, args.worklist_out)
        print(f"工作单 -> {out}：{n} 条，覆盖 {fams} 个 family")
        return

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"标注 UI -> {url}  (Ctrl-C 退出)")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
