#!/usr/bin/env python3
"""Generate a self-contained HTML evaluation dashboard from ``results/summary.json``.

The dashboard compares every agent in the summary (learned PPO/SAC and the
classical PID/Stanley baselines) across the driving metrics, and shows each
agent's return distribution. It embeds the data inline, so the resulting HTML
file is fully portable -- open it in any browser, no server required.

Usage::

    python scripts/make_dashboard.py --results results/summary.json --out dashboard/index.html
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

METRIC_LABELS = {
    "success_rate": "Success rate",
    "mean_route_completion": "Route completion",
    "collision_rate": "Collision rate",
    "offroad_rate": "Off-road rate",
    "mean_return": "Mean return",
    "mean_speed_kmh": "Mean speed (km/h)",
    "mean_abs_lateral_error_m": "Lateral error (m)",
    "mean_abs_jerk": "Steering jerk",
    "mean_episode_length": "Episode length",
}
# Metrics where lower is better (used to colour the table).
LOWER_IS_BETTER = {"collision_rate", "offroad_rate", "mean_abs_lateral_error_m", "mean_abs_jerk"}


def build_html(summary: dict, generated: str) -> str:
    data_json = json.dumps(summary)
    labels_json = json.dumps(METRIC_LABELS)
    lower_json = json.dumps(sorted(LOWER_IS_BETTER))
    # The page logic lives in a template; data is injected as JSON literals.
    return _TEMPLATE.replace("__DATA__", data_json).replace(
        "__LABELS__", labels_json
    ).replace("__LOWER__", lower_json).replace("__GENERATED__", generated)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Autonomous Driving RL — Evaluation Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#0e1116; --card:#171c24; --muted:#8b97a7; --text:#e8edf3; --accent:#4f9cff;
    --learned:#4f9cff; --learned2:#7c5cff; --classical:#36c98d; --classical2:#28b487; --border:#232a34;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;}
  .wrap{max-width:1100px;margin:0 auto;padding:28px 20px 64px}
  header h1{margin:0 0 4px;font-size:24px;letter-spacing:.2px}
  header p{margin:0;color:var(--muted);font-size:13px}
  .grid{display:grid;gap:18px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));margin-top:22px}
  .card{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:18px}
  .card h2{margin:0 0 12px;font-size:14px;font-weight:600;color:var(--text)}
  .full{grid-column:1/-1}
  table{width:100%;border-collapse:collapse;font-size:13px}
  th,td{padding:8px 10px;text-align:right;border-bottom:1px solid var(--border)}
  th:first-child,td:first-child{text-align:left}
  thead th{color:var(--muted);font-weight:600}
  .badge{display:inline-block;font-size:11px;padding:2px 8px;border-radius:999px;margin-left:8px}
  .b-learned{background:rgba(79,156,255,.16);color:#9ec5ff}
  .b-classical{background:rgba(54,201,141,.16);color:#7fe3bd}
  .controls{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
  .controls button{background:#0f141b;border:1px solid var(--border);color:var(--text);
       padding:6px 12px;border-radius:8px;font-size:12px;cursor:pointer}
  .controls button.active{border-color:var(--accent);color:#fff;background:rgba(79,156,255,.14)}
  .legend{font-size:12px;color:var(--muted);margin-top:8px}
  canvas{max-height:320px}
  .foot{margin-top:26px;color:var(--muted);font-size:12px;text-align:center}
  .empty{color:var(--muted);font-size:13px;padding:8px 0}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>🚗 Autonomous Driving with RL — Evaluation Dashboard</h1>
    <p>PPO &amp; SAC vs. classical PID / Stanley baselines · generated __GENERATED__</p>
  </header>

  <div class="grid">
    <div class="card full">
      <h2>Metrics comparison</h2>
      <div class="controls" id="metric-controls"></div>
      <canvas id="barChart"></canvas>
      <div class="legend" id="bar-legend"></div>
    </div>

    <div class="card full">
      <h2>Summary table</h2>
      <div id="table-host"></div>
    </div>

    <div class="card full">
      <h2>Return distribution (per evaluation episode)</h2>
      <canvas id="returnChart"></canvas>
    </div>
  </div>

  <div class="foot">
    Built by <code>scripts/make_dashboard.py</code>. Re-run training then
    <code>make eval &amp;&amp; make dashboard</code> to refresh with your own results.
  </div>
</div>

<script>
const DATA = __DATA__;
const LABELS = __LABELS__;
const LOWER = __LOWER__;

const agents = Object.keys((DATA && DATA.agents) || {});
const palette = {
  PPO:'#4f9cff', SAC:'#7c5cff', PID:'#36c98d', STANLEY:'#f4b740',
};
const isLearned = a => ['PPO','SAC'].includes(a.toUpperCase());
const colorFor = a => palette[a.toUpperCase()] || '#9aa7b8';

function metricValue(agent, key){
  const m = DATA.agents[agent] && DATA.agents[agent].metrics || {};
  return (key in m) ? m[key] : null;
}

// ---- Bar chart with metric switcher ----------------------------------------
const metricKeys = Object.keys(LABELS).filter(k => agents.some(a => metricValue(a,k)!==null));
let activeMetric = metricKeys.includes('success_rate') ? 'success_rate' : metricKeys[0];
let barChart, returnChart;

function renderControls(){
  const host = document.getElementById('metric-controls');
  host.innerHTML = '';
  metricKeys.forEach(k=>{
    const b=document.createElement('button');
    b.textContent=LABELS[k]; b.className = (k===activeMetric?'active':'');
    b.onclick=()=>{activeMetric=k; renderControls(); renderBar();};
    host.appendChild(b);
  });
}

function renderBar(){
  const ctx=document.getElementById('barChart');
  const vals=agents.map(a=>metricValue(a,activeMetric));
  const colors=agents.map(colorFor);
  if(barChart) barChart.destroy();
  barChart=new Chart(ctx,{type:'bar',
    data:{labels:agents,datasets:[{label:LABELS[activeMetric],data:vals,backgroundColor:colors,borderRadius:6}]},
    options:{plugins:{legend:{display:false}},
      scales:{y:{grid:{color:'#222a35'},ticks:{color:'#8b97a7'}},x:{grid:{display:false},ticks:{color:'#cdd6e2'}}}}});
  const lb=LOWER.includes(activeMetric)?'lower is better':'higher is better';
  document.getElementById('bar-legend').textContent = `${LABELS[activeMetric]} — ${lb}`;
}

// ---- Summary table ---------------------------------------------------------
function renderTable(){
  const host=document.getElementById('table-host');
  if(agents.length===0){host.innerHTML='<div class="empty">No results yet. Run <code>make smoke</code> or training, then evaluate.</div>';return;}
  let html='<table><thead><tr><th>Metric</th>';
  agents.forEach(a=>{html+=`<th>${a}<span class="badge ${isLearned(a)?'b-learned':'b-classical'}">${isLearned(a)?'learned':'classical'}</span></th>`;});
  html+='</tr></thead><tbody>';
  metricKeys.forEach(k=>{
    html+=`<tr><td>${LABELS[k]}</td>`;
    const vals=agents.map(a=>metricValue(a,k));
    const valid=vals.filter(v=>v!==null);
    const best = LOWER.includes(k)? Math.min(...valid): Math.max(...valid);
    agents.forEach((a,i)=>{
      const v=vals[i];
      const txt = v===null?'—':(Math.abs(v)>=100?v.toFixed(1):v.toFixed(3));
      const hi = (v!==null && v===best)?'style="color:#fff;font-weight:700"':'';
      html+=`<td ${hi}>${txt}</td>`;
    });
    html+='</tr>';
  });
  html+='</tbody></table>';
  host.innerHTML=html;
}

// ---- Return distribution (mean ± std bars over returns) --------------------
function renderReturns(){
  const ctx=document.getElementById('returnChart');
  const means=[],errs=[],colors=[];
  agents.forEach(a=>{
    const r=(DATA.agents[a].returns)||[];
    if(r.length){const m=r.reduce((x,y)=>x+y,0)/r.length;
      const sd=Math.sqrt(r.reduce((s,y)=>s+(y-m)*(y-m),0)/r.length);
      means.push(m);errs.push(sd);}
    else{means.push(metricValue(a,'mean_return')||0);errs.push(0);}
    colors.push(colorFor(a));
  });
  if(returnChart) returnChart.destroy();
  returnChart=new Chart(ctx,{type:'bar',
    data:{labels:agents,datasets:[{label:'Mean return',data:means,backgroundColor:colors,borderRadius:6}]},
    options:{plugins:{legend:{display:false},
      tooltip:{callbacks:{label:(c)=>`return ${c.parsed.y.toFixed(1)} ± ${errs[c.dataIndex].toFixed(1)}`}}},
      scales:{y:{grid:{color:'#222a35'},ticks:{color:'#8b97a7'}},x:{grid:{display:false},ticks:{color:'#cdd6e2'}}}}});
}

renderControls(); renderBar(); renderTable(); renderReturns();
</script>
</body>
</html>
"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/summary.json")
    parser.add_argument("--out", default="dashboard/index.html")
    args = parser.parse_args(argv)

    results_path = Path(args.results)
    summary = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {"agents": {}}
    generated = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(summary, generated), encoding="utf-8")
    print(f"Dashboard written -> {out_path}  ({len(summary.get('agents', {}))} agents)")


if __name__ == "__main__":
    main()
