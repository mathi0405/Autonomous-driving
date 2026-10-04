"""Offline evidence reports and a timeline-controlled driving replay."""

from __future__ import annotations

import html
import json
from pathlib import Path

from ad_rl.validation.readiness import assess_readiness


def render_report(baseline: dict, candidate: dict, verdict: dict, out: Path) -> None:
    """Generate a portable HTML report with no network dependencies."""
    readiness = assess_readiness(candidate)
    rows = []
    for case in verdict["cases"]:
        rows.append(
            "<tr><td>"
            + html.escape(case["id"])
            + "</td><td>"
            + html.escape(case["baseline_reason"])
            + "</td><td>"
            + html.escape(case["candidate_reason"])
            + "</td><td>"
            + f"{case['route_delta']:+.3f}</td><td>"
            + html.escape(", ".join(case["reason_codes"]) or "—")
            + "</td></tr>"
        )
    reasons = "".join(
        f"<li><b>{html.escape(r['code'])}</b>: {r['observed']:.4g} "
        f"(limit {r['limit']:.4g})</li>"
        for r in verdict["reason_codes"]
    )
    cards = ""
    for title, run in (("Baseline", baseline), ("Candidate", candidate)):
        s = run["summary"]
        cards += (
            f"<article><h2>{title}: {html.escape(run['policy']['name'])}</h2>"
            f"<strong>{s['success_rate']:.1%}</strong> successful episodes"
            f"<p>{s['episodes']} cases · {s['collision_rate']:.1%} collision rate · "
            f"{s['offroad_rate']:.1%} off-road rate</p>"
            f"<p>Success 95% Wilson interval: {s['success_95ci'][0]:.1%} to "
            f"{s['success_95ci'][1]:.1%}</p></article>"
        )
    content = (
        f"<h1>Driving regression check <span>{verdict['verdict']}</span></h1>"
        "<p>Paired scenarios. Reproducible evidence. Explicit acceptance rules.</p>"
        f"<section>{cards}</section>"
        f"<h2>Simulation release: {readiness['status']}</h2>"
        "<p>This separate check requires sufficient cases in every regime, zero failed missions "
        "and bounded peak speed. It does not approve hardware deployment.</p><pre>"
        + html.escape(json.dumps(readiness["blockers"], indent=2))
        + "</pre><h2>Regression reasons</h2><ul>"
        f"{reasons or '<li>No regression threshold exceeded.</li>'}</ul>"
        f"<p>Mean route delta: {verdict['route_delta']:+.4f}; paired bootstrap 95% interval: "
        f"{verdict['route_delta_paired_bootstrap_95ci']}</p>"
        "<h2>Scenario outcomes</h2><table><thead><tr><th>Scenario</th><th>Baseline</th>"
        "<th>Candidate</th><th>Route delta</th><th>Reasons</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table><h2>Acceptance contract</h2><pre>"
        + html.escape(json.dumps(verdict["rules"], indent=2))
        + "</pre>"
        "<h2>Scope</h2><p>CPU kinematic bicycle simulation with synthetic state sensing, "
        "static circular obstacle checks and discrete physics steps. Results do not "
        "establish camera-policy, CARLA, or real-vehicle performance.</p>"
        "<h2>Provenance</h2><pre>"
        + html.escape(json.dumps(candidate["provenance"], indent=2))
        + "</pre>"
    )
    out.write_text(_page(content, "Driving regression report"), encoding="utf-8")


def render_replay(episode: dict, out: Path) -> None:
    """Embed a complete episode, road geometry and scrubber in an offline viewer."""
    data = json.dumps(episode, allow_nan=False).replace("<", "\\u003c")
    body = (
        "<h1>Driving failure replay</h1><p id='label'></p>"
        "<canvas id='scene' width='1200' height='550'></canvas><p>"
        "<button id='play'>Play / pause</button> "
        "<input id='time' type='range' min='0' value='0' style='width:70%'></p>"
        "<pre id='telemetry'></pre><details><summary>Scenario and result</summary><pre>"
        + html.escape(
            json.dumps({"scenario": episode["scenario"], "metrics": episode["metrics"]}, indent=2)
        )
        + "</pre></details><script>const episode="
        + data
        + ";"
        + _REPLAY_JS
        + "</script>"
    )
    out.write_text(_page(body, "Driving replay"), encoding="utf-8")


def _page(body: str, title: str) -> str:
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{title}</title><style>"
        "body{font:16px system-ui;background:#101923;color:#e4edf5;max-width:1200px;"
        "margin:auto;padding:32px}h1{font-size:34px}h2{font-size:20px}"
        "section{display:flex;gap:20px;flex-wrap:wrap}article{background:#1b2937;"
        "padding:24px;border-radius:12px;flex:1}strong{font-size:32px;color:#6cddbd}"
        "span{color:#f3be64}table{width:100%;border-collapse:collapse}"
        "td,th{text-align:left;padding:10px;border-bottom:1px solid #334557}"
        "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#1b2937;padding:16px}"
        "canvas{width:100%;background:#14212d;border-radius:12px}"
        "button{padding:10px;border:0;border-radius:8px;cursor:pointer}"
        "</style>" + body + "</html>"
    )


_REPLAY_JS = r"""
const canvas=document.getElementById('scene'), ctx=canvas.getContext('2d');
const slider=document.getElementById('time'), points=episode.trajectory;
slider.max=points.length-1;
let playing=false,previous=0;
const road=episode.scene.road, all=road.concat(points.map(p=>[p.x_m,p.y_m]));
const xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);
const xmin=Math.min(...xs)-5,xmax=Math.max(...xs)+5;
const ymin=Math.min(...ys)-5,ymax=Math.max(...ys)+5;
const scale=Math.min(1140/(xmax-xmin),490/(ymax-ymin));
function pixel(x,y){return [30+(x-xmin)*scale,520-(y-ymin)*scale];}
function line(arr,color,width){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();
 arr.forEach((p,i)=>{let q=pixel(p[0],p[1]);i?ctx.lineTo(...q):ctx.moveTo(...q)});ctx.stroke();}
function draw(){let n=Number(slider.value),p=points[n];ctx.clearRect(0,0,1200,550);
 line(road,'#435467',2*episode.scene.road_half_width_m*scale);
 line(road,'#b7c6d3',1);
 episode.scene.obstacles.forEach(o=>{let q=pixel(...o);ctx.fillStyle='#ef785e';
 ctx.beginPath();ctx.arc(...q,episode.scene.collision_radius_m*scale,0,Math.PI*2);ctx.fill();});
 line(points.slice(0,n+1).map(t=>[t.x_m,t.y_m]),'#65dfbb',3);
 let q=pixel(p.x_m,p.y_m);ctx.save();ctx.translate(...q);ctx.rotate(-p.yaw_rad);
 ctx.fillStyle=p.reason==='COLLISION'||p.reason==='OFFROAD'?'#ff615c':'#65dfbb';
 ctx.fillRect(-5,-3,10,6);ctx.restore();
 document.getElementById('label').textContent=episode.scenario.id+' · '
 +episode.policy.name+' · '+episode.metrics.terminal_reason;
 document.getElementById('telemetry').textContent='t = '+p.time_s.toFixed(1)
 +' s | speed = '+(p.speed_ms*3.6).toFixed(1)+' km/h | lateral error = '
 +p.lateral_error_m.toFixed(3)+' m | clearance = '+p.minimum_clearance_m.toFixed(3)
 +' m\nRequested action: '+JSON.stringify(p.requested_action)+' | applied: '
 +JSON.stringify(p.applied_action)+' | '+p.reason;}
slider.oninput=draw;document.getElementById('play').onclick=()=>{playing=!playing;};
function animate(t){if(playing&&t-previous>60){
 slider.value=(Number(slider.value)+1)%points.length;draw();previous=t;}
 requestAnimationFrame(animate);}
draw();requestAnimationFrame(animate);
"""
