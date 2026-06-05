"""
Real-time training dashboard. Reads TensorBoard event logs and serves
a self-refreshing HTML page with loss curves and training speed.

Usage:
    python scripts/dashboard.py --log_dir checkpoints/logs/ --port 6008
"""
import argparse
import json
import struct
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TEMPLATE = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f0f0f;color:#e0e0e0;font:14px system-ui,sans-serif;padding:20px}
.header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
.header h1{font-size:20px;color:#fff}
.status{display:flex;gap:20px;font-size:13px}
.status .val{color:#4fc3f7;font-weight:700}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.card{background:#1a1a1a;border-radius:8px;padding:16px}
.card h2{font-size:14px;color:#888;margin-bottom:10px}
.chart{position:relative;height:220px}
canvas{width:100%!important}
.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}
.kpi{background:#1a1a1a;border-radius:8px;padding:14px;text-align:center}
.kpi .label{font-size:12px;color:#888}
.kpi .value{font-size:28px;font-weight:700;color:#fff;margin:4px 0}
.kpi .unit{font-size:11px;color:#666}
#footer{text-align:center;color:#555;font-size:11px;margin-top:8px}
</style>
</head><body>
<div class="header">
<h1>🎵 Neural Pitch Corrector — Training</h1>
<div class="status">
<span>Epoch: <span class="val" id="epoch">-</span></span>
<span>Step: <span class="val" id="step">-</span></span>
<span>Speed: <span class="val" id="speed">-</span></span>
<span>Updated: <span class="val" id="updated">-</span></span>
</div>
</div>
<div class="summary">
<div class="kpi"><div class="label">G Loss</div><div class="value" id="kpi_g">-</div><div class="unit">generator</div></div>
<div class="kpi"><div class="label">D Loss</div><div class="value" id="kpi_d">-</div><div class="unit">discriminator</div></div>
<div class="kpi"><div class="label">Mel Loss</div><div class="value" id="kpi_mel">-</div><div class="unit">mel-spectrogram</div></div>
<div class="kpi"><div class="label">LR</div><div class="value" id="kpi_lr">-</div><div class="unit">learning rate</div></div>
</div>
<div class="grid">
<div class="card"><h2>Generator & Discriminator Loss</h2><div class="chart"><canvas id="loss_chart"></canvas></div></div>
<div class="card"><h2>Mel-Spectrogram & Feature Matching Loss</h2><div class="chart"><canvas id="mel_chart"></canvas></div></div>
<div class="card"><h2>Adversarial Loss</h2><div class="chart"><canvas id="adv_chart"></canvas></div></div>
<div class="card"><h2>Training Speed & Latency</h2><div class="chart"><canvas id="speed_chart"></canvas></div></div>
</div>
<div id="footer">Auto-refreshing every 10s</div>
<script>
const COLORS={g:'#4fc3f7',d:'#ef5350',mel:'#66bb6a',fm:'#ffa726',adv:'#ab47bc',io:'#78909c',hf:'#b0bec5',dc:'#ef5350',gn:'#4fc3f7'};
function fmt(v,d=2){return v===null||v===undefined?'-':Number(v).toFixed(d)}
function ts(){return new Date().toLocaleTimeString()}

let charts={};
['loss','mel','adv','speed'].forEach(k=>{
    const ctx=document.getElementById(k+'_chart').getContext('2d');
    charts[k]=new Chart(ctx,{
        type:'line',options:{responsive:true,maintainAspectRatio:false,
            animation:false,
            plugins:{legend:{labels:{color:'#999',font:{size:10}}}},
            scales:{x:{ticks:{color:'#666',maxTicksLimit:8},grid:{color:'#222'}},
                    y:{ticks:{color:'#666',maxTicksLimit:6},grid:{color:'#222'}}}},
        data:{labels:[],datasets:[]}
    });
});

function makeDs(label,color,data){
    return {label:label,borderColor:color,backgroundColor:color+'33',data:data,
            pointRadius:0,borderWidth:1.5,tension:0.3,fill:false}
}

async function fetchData(){
    try{
        const r=await fetch('/data');
        const d=await r.json();

        // KPIs
        const lg=d.series['train/loss_g'], ld=d.series['train/loss_d'],
              lm=d.series['train/loss_mel'], ll=d.series['train/lr'];
        document.getElementById('kpi_g').textContent=lg?fmt(lg.values.at(-1)?.[1]):'-';
        document.getElementById('kpi_d').textContent=ld?fmt(ld.values.at(-1)?.[1]):'-';
        document.getElementById('kpi_mel').textContent=lm?fmt(lm.values.at(-1)?.[1],1):'-';
        document.getElementById('kpi_lr').textContent=ll?ll.values.at(-1)?.[1]?.toExponential(2):'-';
        document.getElementById('epoch').textContent=d.epoch||'-';
        document.getElementById('step').textContent=d.last_step||'-';

        // Speed estimate
        const st=d.series['profile/gen_ms']; const ss=d.last_step;
        if(st&&st.values.length>1){
            const recent=st.values.slice(-50);
            const avgMs=recent.reduce((s,v)=>s+v[1],0)/recent.length;
            document.getElementById('speed').textContent=(1000/avgMs).toFixed(1)+' it/s';
        }
        document.getElementById('updated').textContent=ts();

        // Update charts
        const steps=(d.series['train/loss_g']||{}).values||[];
        const labels=steps.map(s=>s[0]);

        function ds(key,color){return makeDs(key,color,(d.series[key]||{}).values||[])}
        charts.loss.data={labels,datasets:[ds('train/loss_g','#4fc3f7'),ds('train/loss_d','#ef5350')]};
        charts.mel.data={labels,datasets:[ds('train/loss_mel','#66bb6a'),ds('train/loss_fm','#ffa726')]};
        charts.adv.data={labels,datasets:[ds('train/loss_adv','#ab47bc')]};
        charts.speed.data={labels,datasets:[
            ds('profile/gen_ms','#4fc3f7'),ds('profile/disc_ms','#ef5350'),
            ds('profile/io_ms','#78909c'),ds('profile/mel_ms','#b0bec5')]};

        Object.values(charts).forEach(c=>c.update('none'));
    }catch(e){console.error(e)}
}
fetchData();setInterval(fetchData,10000);
</script>
</body></html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    accumulator = None

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/data":
            self._serve_data()
        else:
            self.send_error(404)

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(TEMPLATE.encode())

    def _serve_data(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        data = {"series": {}, "epoch": 0, "last_step": 0}
        if self.accumulator:
            self.accumulator.Reload()
            tags = self.accumulator.Tags().get("scalars", [])
            for tag in tags:
                events = self.accumulator.Scalars(tag)
                data["series"][tag] = {
                    "values": [[e.step, e.value] for e in events[-2000:]]
                }
            if data["series"]:
                all_steps = [v["values"] for v in data["series"].values()]
                if all_steps:
                    max_step = max(v[-1][0] for v in all_steps if v)
                    data["last_step"] = max_step

        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass  # suppress access logs


def main():
    parser = argparse.ArgumentParser(description="Training dashboard server")
    parser.add_argument("--log_dir", default="checkpoints/logs/")
    parser.add_argument("--port", type=int, default=6008)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    print(f"Watching: {log_dir}")
    print(f"Dashboard: http://{args.host}:{args.port}")

    # Wait for TF events to appear
    for _ in range(30):
        tf_files = list(log_dir.rglob("events.out.tfevents.*"))
        if tf_files:
            break
        print("  Waiting for TensorBoard events...")
        time.sleep(2)

    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 10000})
    acc.Reload()
    DashboardHandler.accumulator = acc

    server = HTTPServer((args.host, args.port), DashboardHandler)
    print(f"Ready → http://localhost:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    main()
