#!/usr/bin/env python3
"""Generate an interactive HTML chart from sweep results (Chart.js, no deps)."""

import json, sys
from pathlib import Path


def main(result_dir: str) -> None:
    out = Path(result_dir)
    summary = json.loads((out / "summary.json").read_text())

    qps = [e["params"]["qp"] for e in summary]
    bw = [e["BW_average"] for e in summary]
    rates = [e["MsgRate"] for e in summary]

    # Perf symbols per combo
    perf_data = []
    for i in range(len(summary)):
        p = out / f"{i+1:04d}" / "result.json"
        if p.exists():
            perf_data.append(json.loads(p.read_text())["_process"]["server_perf"])
        else:
            perf_data.append({})

    # Top symbols across all combos
    all_syms = set()
    for pd in perf_data:
        for s, v in sorted(pd.items(), key=lambda x: -x[1])[:5]:
            if v > 0:
                all_syms.add(s)
    sym_total = {s: sum(pd.get(s, 0) for pd in perf_data) for s in all_syms}
    top_syms = sorted(sym_total, key=lambda s: -sym_total[s])[:7]

    colors = [
        "#2563eb", "#dc2626", "#16a34a", "#ca8a04",
        "#9333ea", "#0891b2", "#be123c", "#d1d5db",
    ]
    datasets = []
    for idx, sym in enumerate(top_syms):
        vals = [pd.get(sym, 0) for pd in perf_data]
        datasets.append(
            json.dumps({
                "label": sym[:35], "data": vals,
                "backgroundColor": colors[idx % len(colors)],
            })
        )
    other_vals = [
        max(0, 100 - sum(pd.get(s, 0) for s in top_syms))
        for pd in perf_data
    ]
    datasets.append(
        json.dumps({
            "label": "(other)", "data": other_vals,
            "backgroundColor": "#d1d5db",
        })
    )

    cores = sorted(
        [
            k
            for k in summary[0]["cpu_per_core"]
            if k.startswith("cpu") and k != "cpu"
        ],
        key=lambda c: int(c.replace("cpu", "")),
    )
    cpu_rows = []
    for e in summary:
        cpu_rows.append([round(e["cpu_per_core"][c], 1) for c in cores])

    _labels = json.dumps([str(q) for q in qps])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>RDMA Sweep Results</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f8fafc; }}
  h1 {{ font-size: 1.3rem; margin-bottom: 4px; }}
  .sub {{ color: #64748b; font-size: .85rem; margin-bottom: 20px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .card {{ background: #fff; border-radius: 12px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.1); }}
  .card h2 {{ font-size: .95rem; margin: 0 0 8px 0; color: #334155; }}
  canvas {{ max-height: 320px; }}
  table {{ font-size: .8rem; border-collapse: collapse; width: 100%; }}
  th, td {{ padding: 3px 6px; text-align: center; border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f1f5f9; font-weight: 600; }}
</style></head><body>
<h1>RDMA Write BW Sweep</h1>
<p class="sub">SoftRoCE (rxe0) · 64K msg · ib_write_bw · server perf record -g</p>
<div class="grid">
  <div class="card"><h2>Bandwidth (MB/s)</h2><canvas id="bwChart"></canvas></div>
  <div class="card"><h2>Message Rate (Mmsg/s)</h2><canvas id="rateChart"></canvas></div>
  <div class="card" style="grid-column:1/3"><h2>Top CPU Consumers (self %)</h2><canvas id="perfChart"></canvas></div>
  <div class="card" style="grid-column:1/3"><h2>Per-Core CPU Utilization (%)</h2><canvas id="cpuChart"></canvas></div>
</div>
<script>
new Chart(document.getElementById('bwChart'),{{type:'line',data:{{labels:{_labels},datasets:[{{label:'BW',data:{json.dumps(bw)},borderColor:'#2563eb',backgroundColor:'rgba(37,99,235,.1)',fill:true,tension:.2,pointRadius:5}}]}},options:{{responsive:true,scales:{{x:{{title:{{display:true,text:'Queue Pairs'}},type:'logarithmic'}},y:{{title:{{display:true,text:'MB/s'}},beginAtZero:true}}}}}}}});
new Chart(document.getElementById('rateChart'),{{type:'line',data:{{labels:{_labels},datasets:[{{label:'MsgRate',data:{json.dumps(rates)},borderColor:'#16a34a',backgroundColor:'rgba(22,163,74,.1)',fill:true,tension:.2,pointRadius:5}}]}},options:{{responsive:true,scales:{{x:{{title:{{display:true,text:'Queue Pairs'}},type:'logarithmic'}},y:{{title:{{display:true,text:'Mmsg/s'}},beginAtZero:true}}}}}}}});
new Chart(document.getElementById('perfChart'),{{type:'bar',data:{{labels:{_labels},datasets:[{','.join(datasets)}]}},options:{{responsive:true,scales:{{x:{{stacked:true,title:{{display:true,text:'Queue Pairs'}}}},y:{{stacked:true,max:100,title:{{display:true,text:'Self %'}}}}}},plugins:{{legend:{{position:'bottom',labels:{{boxWidth:12,font:{{size:10}}}}}}}}}}}});
const cd={json.dumps(cpu_rows)},cl={json.dumps(cores)},ql={_labels};
const bg=v=>v>20?'#dc2626':v>10?'#f59e0b':v>3?'#eab308':'#e2e8f0';
const tc=v=>v>20?'#fff':'#1e293b';
const tbl=document.getElementById('cpuChart').parentNode;
const t=document.createElement('table');
let h='<tr><th>QP</th>'+cl.map(c=>'<th>'+c+'</th>').join('')+'</tr>';
cd.forEach((r,i)=>{{h+='<tr><td><b>'+ql[i]+'</b></td>';r.forEach(v=>{{h+='<td style="background:'+bg(v)+';color:'+tc(v)+'">'+v+'</td>'}});h+='</tr>'}});
t.innerHTML=h;tbl.appendChild(t);
</script></body></html>"""
    (out / "chart.html").write_text(html)
    print(f"chart -> {out / 'chart.html'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
