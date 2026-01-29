from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import pandas as pd
import joblib

# -----------------------------
# Load models (make sure these files exist in same folder)
# -----------------------------
PRIORITY_MODEL_PATH = "priority_model_tirana_v2.joblib"
SEVERITY_MODEL_PATH = "severity_model_tirana_v2.joblib"

priority_model = joblib.load(PRIORITY_MODEL_PATH)
severity_model = joblib.load(SEVERITY_MODEL_PATH)

app = FastAPI(
    title="PAWSIGHT API",
    version="1.0",
    description="AI-driven animal incident triage and severity scoring"
)



# -----------------------------
# Input schema (MUST match training features)
# -----------------------------
class IncidentInput(BaseModel):
    neighborhood: str
    animal_type: str
    age_group: str
    behavior: str
    mobility: str
    traffic_risk: str
    weather: str
    report_channel: str
    visible_injury: str

    animal_count: int = Field(ge=1, le=10)

    near_traffic: int = Field(ge=0, le=1)
    near_school: int = Field(ge=0, le=1)
    very_thin: int = Field(ge=0, le=1)
    bleeding: int = Field(ge=0, le=1)
    open_wound: int = Field(ge=0, le=1)
    pregnant_or_nursing: int = Field(ge=0, le=1)
    has_collar: int = Field(ge=0, le=1)

    reporter_confidence: int = Field(ge=1, le=5)

    report_hour: int = Field(ge=0, le=23)
    report_dayofweek: int = Field(ge=0, le=6)
    report_month: int = Field(ge=1, le=12)


FEATURE_COLS = [
    "neighborhood", "animal_type", "age_group",
    "behavior", "mobility", "traffic_risk",
    "weather", "report_channel", "visible_injury",
    "animal_count",
    "near_traffic", "near_school", "very_thin", "bleeding", "open_wound",
    "pregnant_or_nursing", "has_collar", "reporter_confidence",
    "report_hour", "report_dayofweek", "report_month"
]


def to_dataframe(payload: dict) -> pd.DataFrame:
    # Ensure column order exactly matches training
    df = pd.DataFrame([payload])
    return df[FEATURE_COLS]


# -----------------------------
# API endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(inp: IncidentInput):
    X = to_dataframe(inp.model_dump())
    pred_priority = priority_model.predict(X)[0]
    pred_severity = float(severity_model.predict(X)[0])

    # Keep severity inside 0-100 range for display
    pred_severity = max(0.0, min(100.0, pred_severity))

    # If classifier supports probabilities, return them too (nice for demo)
    probs = None
    if hasattr(priority_model, "predict_proba"):
        p = priority_model.predict_proba(X)[0]
        classes = list(priority_model.named_steps["model"].classes_)
        probs = {cls: float(p[i]) for i, cls in enumerate(classes)}

    return {
        "priority_prediction": pred_priority,
        "severity_prediction": round(pred_severity, 2),
        "priority_probabilities": probs
    }


# -----------------------------
# Simple Web UI (test page)
# -----------------------------
from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>PAWSIGHT — Animal Triage System</title>

  <style>
    :root{
      --bg:#0b1220;
      --panel:#0f1a2e;
      --card:#111f36;
      --muted:#a9b4c7;
      --text:#e9eef8;
      --border:rgba(255,255,255,.10);
      --shadow: 0 20px 50px rgba(0,0,0,.35);
      --radius:18px;

      --p1:#ff4d4d;
      --p2:#ffb020;
      --p3:#41d18a;
      --accent:#7aa7ff;
    }

    *{ box-sizing:border-box; }
    body{
      margin:0;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, "Apple Color Emoji","Segoe UI Emoji";
      background: radial-gradient(1200px 700px at 20% 0%, #15264a 0%, var(--bg) 55%);
      color:var(--text);
      padding:32px 18px 60px;
    }

    .wrap{ max-width:1100px; margin:0 auto; }

    .top{
      display:flex; align-items:flex-start; justify-content:space-between;
      gap:16px; margin-bottom:18px;
    }
    h1{
      font-size:28px; margin:0 0 6px; letter-spacing:.2px;
    }
    .sub{
      margin:0; color:var(--muted); line-height:1.4;
      max-width:760px;
    }

    .actions{
      display:flex; gap:10px; flex-wrap:wrap; justify-content:flex-end;
    }
    .btn{
      background: rgba(255,255,255,.06);
      border:1px solid var(--border);
      color:var(--text);
      padding:10px 12px;
      border-radius:12px;
      cursor:pointer;
      transition: transform .08s ease, background .2s ease;
      font-weight:600;
    }
    .btn:hover{ background: rgba(255,255,255,.10); }
    .btn:active{ transform: scale(.98); }
    .btn.primary{
      background: linear-gradient(135deg, rgba(122,167,255,.35), rgba(122,167,255,.12));
      border-color: rgba(122,167,255,.35);
    }

    .grid{
      display:grid;
      grid-template-columns: 1.2fr .8fr;
      gap:18px;
      margin-top:18px;
    }

    .card{
      background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
      border:1px solid var(--border);
      border-radius:var(--radius);
      box-shadow: var(--shadow);
      overflow:hidden;
    }
    .card-h{
      padding:16px 18px;
      border-bottom:1px solid var(--border);
      background: rgba(0,0,0,.10);
      display:flex; align-items:center; justify-content:space-between; gap:12px;
    }
    .card-h h2{ margin:0; font-size:16px; letter-spacing:.2px; }
    .hint{ color:var(--muted); font-size:12px; }

    .card-b{ padding:16px 18px 18px; }

    .section{
      margin:14px 0 4px;
      font-size:12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .12em;
    }

    .form{
      display:grid;
      grid-template-columns: 1fr 1fr;
      gap:12px;
    }
    .form .full{ grid-column:1 / -1; }

    label{ display:block; font-size:12px; color:var(--muted); margin:0 0 6px; }
    input, select{
      width:100%;
      padding:10px 11px;
      border-radius:12px;
      border:1px solid var(--border);
      background: rgba(15,26,46,.6);
      color: var(--text);
      outline:none;
    }
    input::placeholder{ color: rgba(233,238,248,.45); }
    input:focus, select:focus{
      border-color: rgba(122,167,255,.55);
      box-shadow: 0 0 0 3px rgba(122,167,255,.15);
    }
    .smallrow{
      display:grid;
      grid-template-columns: repeat(4, 1fr);
      gap:12px;
    }

    .footerbar{
      display:flex; align-items:center; justify-content:space-between;
      gap:12px;
      margin-top:14px;
      padding-top:14px;
      border-top:1px solid var(--border);
    }

    .pill{
      display:inline-flex; align-items:center; gap:8px;
      padding:8px 10px;
      border-radius:999px;
      border:1px solid var(--border);
      background: rgba(255,255,255,.05);
      color: var(--muted);
      font-size:12px;
    }
    .dot{ width:10px; height:10px; border-radius:999px; background: var(--accent); }

    /* Results */
    .badge{
      display:inline-flex; align-items:center; gap:10px;
      padding:10px 12px;
      border-radius:14px;
      font-weight:800;
      letter-spacing:.3px;
      border:1px solid var(--border);
      background: rgba(255,255,255,.06);
    }
    .badge .dot{ width:12px; height:12px; }
    .badge.p1 .dot{ background: var(--p1); }
    .badge.p2 .dot{ background: var(--p2); }
    .badge.p3 .dot{ background: var(--p3); }

    .kpi{
      display:grid;
      grid-template-columns: 1fr 1fr;
      gap:12px;
      margin-top:12px;
    }
    .kpi .box{
      border:1px solid var(--border);
      border-radius:14px;
      padding:12px;
      background: rgba(255,255,255,.04);
    }
    .kpi .lab{ font-size:12px; color:var(--muted); margin-bottom:6px; }
    .kpi .val{ font-size:20px; font-weight:900; }

    .barwrap{
      margin-top:12px;
      border:1px solid var(--border);
      border-radius:14px;
      background: rgba(255,255,255,.04);
      padding:12px;
    }
    .barhead{
      display:flex; justify-content:space-between; align-items:center;
      gap:10px;
      margin-bottom:10px;
    }
    .barhead .t{ font-size:12px; color:var(--muted); }
    .bar{
      width:100%;
      height:10px;
      border-radius:999px;
      background: rgba(255,255,255,.08);
      overflow:hidden;
      border:1px solid rgba(255,255,255,.10);
    }
    .bar > div{
      height:100%;
      width:0%;
      background: linear-gradient(90deg, rgba(65,209,138,.9), rgba(255,176,32,.9), rgba(255,77,77,.9));
      border-radius:999px;
      transition: width .35s ease;
    }

    .probs{
      margin-top:12px;
      border:1px solid var(--border);
      border-radius:14px;
      overflow:hidden;
      background: rgba(255,255,255,.04);
    }
    .probs .r{
      display:flex; align-items:center; justify-content:space-between;
      padding:10px 12px;
      border-bottom:1px solid rgba(255,255,255,.08);
      font-size:13px;
    }
    .probs .r:last-child{ border-bottom:none; }
    .tag{
      padding:6px 10px;
      border-radius:999px;
      border:1px solid rgba(255,255,255,.10);
      background: rgba(255,255,255,.05);
      font-weight:800;
      letter-spacing:.2px;
    }

    .err{
      margin-top:12px;
      padding:12px;
      border-radius:14px;
      border:1px solid rgba(255,77,77,.35);
      background: rgba(255,77,77,.10);
      color: #ffd4d4;
      display:none;
      white-space: pre-wrap;
    }

    .note{
      margin-top:12px;
      color: var(--muted);
      font-size:12px;
      line-height:1.4;
    }

    @media (max-width: 920px){
      .grid{ grid-template-columns: 1fr; }
      .actions{ justify-content:flex-start; }
    }
    @media (max-width: 640px){
      .form{ grid-template-columns: 1fr; }
      .smallrow{ grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="top">
      <div>
        <h1>PAWSIGHT</h1>
        <p class="sub">AI-powered animal triage & severity assessment</p>
        <p class="sub" style="margin-top:8px;">
          Fill the incident details and click <b>Predict</b>. Quick Fill buttons are available for demo scenarios.
        </p>
      </div>

      <div class="actions">
        <button class="btn" onclick="fillLow()">Quick Fill: Low</button>
        <button class="btn" onclick="fillMed()">Quick Fill: Medium</button>
        <button class="btn" onclick="fillHigh()">Quick Fill: High</button>
        <button class="btn primary" onclick="predict()">Predict</button>
      </div>
    </div>

    <div class="grid">
      <!-- INPUT CARD -->
      <div class="card">
        <div class="card-h">
          <h2>Incident Input</h2>
          <div class="hint">All fields reflect report-time information (no leakage).</div>
        </div>
        <div class="card-b">

          <div class="section">Context</div>
          <div class="form">
            <div>
              <label>Neighborhood</label>
              <input id="neighborhood" placeholder="e.g. Kombinat"/>
            </div>
            <div>
              <label>Animal Type</label>
              <select id="animal_type">
                <option value="">Select…</option>
                <option>Dog</option><option>Cat</option>
              </select>
            </div>
            <div>
              <label>Age Group</label>
              <select id="age_group">
                <option value="">Select…</option>
                <option>Puppy/Kitten</option><option>Adult</option><option>Senior</option><option>Unknown</option>
              </select>
            </div>
            <div>
              <label>Report Channel</label>
              <select id="report_channel">
                <option value="">Select…</option>
                <option>Call</option><option>WhatsApp</option><option>App</option><option>Walk-in</option><option>Email</option>
              </select>
            </div>
          </div>

          <div class="section">Animal Condition</div>
          <div class="form">
            <div>
              <label>Behavior</label>
              <select id="behavior">
                <option value="">Select…</option>
                <option>Calm</option><option>Fearful</option><option>Aggressive</option><option>Weak</option><option>Unknown</option>
              </select>
            </div>
            <div>
              <label>Mobility</label>
              <select id="mobility">
                <option value="">Select…</option>
                <option>Normal</option><option>Limping</option><option>Immobile</option><option>Unknown</option>
              </select>
            </div>
            <div>
              <label>Visible Injury</label>
              <select id="visible_injury">
                <option value="">Select…</option>
                <option>Yes</option><option>No</option>
              </select>
            </div>
            <div>
              <label>Animal Count</label>
              <input id="animal_count" type="number" min="1" max="10" placeholder="1–10"/>
            </div>
          </div>

          <div class="section">Signals (0/1 flags)</div>
          <div class="smallrow">
            <div><label>Near Traffic</label><input id="near_traffic" type="number" min="0" max="1" placeholder="0 or 1"/></div>
            <div><label>Near School</label><input id="near_school" type="number" min="0" max="1" placeholder="0 or 1"/></div>
            <div><label>Very Thin</label><input id="very_thin" type="number" min="0" max="1" placeholder="0 or 1"/></div>
            <div><label>Bleeding</label><input id="bleeding" type="number" min="0" max="1" placeholder="0 or 1"/></div>
            <div><label>Open Wound</label><input id="open_wound" type="number" min="0" max="1" placeholder="0 or 1"/></div>
            <div><label>Pregnant/Nursing</label><input id="pregnant_or_nursing" type="number" min="0" max="1" placeholder="0 or 1"/></div>
            <div><label>Has Collar</label><input id="has_collar" type="number" min="0" max="1" placeholder="0 or 1"/></div>
            <div><label>Reporter Confidence (1–5)</label><input id="reporter_confidence" type="number" min="1" max="5" placeholder="1–5"/></div>
          </div>

          <div class="section">Environment & Time</div>
          <div class="form">
            <div>
              <label>Traffic Risk</label>
              <select id="traffic_risk">
                <option value="">Select…</option>
                <option>Low</option><option>Medium</option><option>High</option>
              </select>
            </div>
            <div>
              <label>Weather</label>
              <select id="weather">
                <option value="">Select…</option>
                <option>Clear</option><option>Rain</option><option>Heatwave</option><option>Cold</option>
              </select>
            </div>

            <div>
              <label>Report Hour (0–23)</label>
              <input id="report_hour" type="number" min="0" max="23" placeholder="0–23"/>
            </div>
            <div>
              <label>Day of Week (0–6)</label>
              <input id="report_dayofweek" type="number" min="0" max="6" placeholder="0–6"/>
            </div>
            <div>
              <label>Month (1–12)</label>
              <input id="report_month" type="number" min="1" max="12" placeholder="1–12"/>
            </div>

            <div class="full">
              <div class="footerbar">
                <span class="pill"><span class="dot"></span>Tip: Quick Fill is great for demos.</span>
                <button class="btn primary" type="button" onclick="predict()">Predict</button>
              </div>
              <div class="err" id="err"></div>
            </div>
          </div>

        </div>
      </div>

      <!-- RESULTS CARD -->
      <div class="card">
        <div class="card-h">
          <h2>Prediction Output</h2>
          <div class="hint">Model output + probabilities.</div>
        </div>
        <div class="card-b">
          <div id="badge" class="badge p3">
            <span class="dot"></span>
            <span id="priority_text">Priority: —</span>
          </div>

          <div class="kpi">
            <div class="box">
              <div class="lab">Severity Score</div>
              <div class="val" id="severity_val">—</div>
            </div>
            <div class="box">
              <div class="lab">Recommended Action</div>
              <div class="val" id="action_val" style="font-size:14px; font-weight:800;">—</div>
            </div>
          </div>

          <div class="barwrap">
            <div class="barhead">
              <div class="t">Severity Meter</div>
              <div class="t"><span id="sev_pct">0</span>%</div>
            </div>
            <div class="bar"><div id="sev_bar"></div></div>
          </div>

          <div class="probs" id="probs">
            <div class="r"><span>P1 probability</span> <span class="tag" id="p1p">—</span></div>
            <div class="r"><span>P2 probability</span> <span class="tag" id="p2p">—</span></div>
            <div class="r"><span>P3 probability</span> <span class="tag" id="p3p">—</span></div>
          </div>

          <div class="note">
            <b>Note:</b> This is a decision-support demo. Outputs should be reviewed by a human dispatcher.
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
  function num(id){
    const v = document.getElementById(id).value;
    return v === "" ? null : Number(v);
  }
  function str(id){ return document.getElementById(id).value; }

  function payload(){
    return {
      neighborhood: str("neighborhood"),
      animal_type: str("animal_type"),
      age_group: str("age_group"),
      behavior: str("behavior"),
      mobility: str("mobility"),
      traffic_risk: str("traffic_risk"),
      weather: str("weather"),
      report_channel: str("report_channel"),
      visible_injury: str("visible_injury"),
      animal_count: num("animal_count"),
      near_traffic: num("near_traffic"),
      near_school: num("near_school"),
      very_thin: num("very_thin"),
      bleeding: num("bleeding"),
      open_wound: num("open_wound"),
      pregnant_or_nursing: num("pregnant_or_nursing"),
      has_collar: num("has_collar"),
      reporter_confidence: num("reporter_confidence"),
      report_hour: num("report_hour"),
      report_dayofweek: num("report_dayofweek"),
      report_month: num("report_month")
    };
  }

  function showError(msg){
    const e = document.getElementById("err");
    e.style.display = "block";
    e.textContent = msg;
  }
  function clearError(){
    const e = document.getElementById("err");
    e.style.display = "none";
    e.textContent = "";
  }

  function clamp01Fields(){
    const ids = ["near_traffic","near_school","very_thin","bleeding","open_wound","pregnant_or_nursing","has_collar"];
    for(const id of ids){
      const el = document.getElementById(id);
      if(el.value === "") continue;
      const v = Number(el.value);
      if(!(v===0 || v===1)) el.value = (v>0 ? 1 : 0);
    }
  }

  function recAction(priority, severity){
    if(priority === "P1" || severity >= 75) return "Dispatch immediately";
    if(priority === "P2" || severity >= 45) return "Respond soon / assess";
    return "Monitor / schedule follow-up";
  }

  function setBadge(priority){
    const b = document.getElementById("badge");
    b.classList.remove("p1","p2","p3");
    if(priority === "P1") b.classList.add("p1");
    else if(priority === "P2") b.classList.add("p2");
    else b.classList.add("p3");
    document.getElementById("priority_text").textContent = "Priority: " + priority;
  }

  function setSeverity(sev){
    const v = Math.max(0, Math.min(100, sev));
    document.getElementById("severity_val").textContent = v.toFixed(2);
    document.getElementById("sev_pct").textContent = Math.round(v);
    document.getElementById("sev_bar").style.width = v + "%";
  }

  function setProbs(probs){
    if(!probs){
      document.getElementById("p1p").textContent = "n/a";
      document.getElementById("p2p").textContent = "n/a";
      document.getElementById("p3p").textContent = "n/a";
      return;
    }
    document.getElementById("p1p").textContent = (probs["P1"]*100).toFixed(1) + "%";
    document.getElementById("p2p").textContent = (probs["P2"]*100).toFixed(1) + "%";
    document.getElementById("p3p").textContent = (probs["P3"]*100).toFixed(1) + "%";
  }

  function validateRequired(p){
    const required = ["neighborhood","animal_type","age_group","behavior","mobility","traffic_risk","weather","report_channel","visible_injury"];
    for(const k of required){
      if(!p[k] || String(p[k]).trim() === ""){
        showError("Please complete all required fields (including dropdowns) before predicting.");
        return false;
      }
    }
    const nums = [
      "animal_count","near_traffic","near_school","very_thin","bleeding","open_wound",
      "pregnant_or_nursing","has_collar","reporter_confidence","report_hour","report_dayofweek","report_month"
    ];
    for(const k of nums){
      if(p[k] === null || Number.isNaN(p[k])){
        showError("Please fill all numeric fields (including 0/1 flags) before predicting.");
        return false;
      }
    }
    return true;
  }

  async function predict(){
    clearError();
    clamp01Fields();

    const p = payload();
    if(!validateRequired(p)) return;

    if(p.reporter_confidence < 1 || p.reporter_confidence > 5){
      showError("Reporter Confidence must be between 1 and 5.");
      return;
    }

    try{
      const res = await fetch("/predict", {
        method:"POST",
        headers:{ "Content-Type":"application/json" },
        body: JSON.stringify(p)
      });

      if(!res.ok){
        const txt = await res.text();
        showError("API error (" + res.status + "):\\n" + txt);
        return;
      }

      const out = await res.json();
      setBadge(out.priority_prediction);
      setSeverity(out.severity_prediction);
      document.getElementById("action_val").textContent = recAction(out.priority_prediction, out.severity_prediction);
      setProbs(out.priority_probabilities);

    }catch(err){
      showError("Could not reach the API. Is the server running?\\n" + err);
    }
  }

  // Quick Fill presets
  function fillLow(){
    document.getElementById("neighborhood").value="Kombinat";
    document.getElementById("animal_type").value="Cat";
    document.getElementById("age_group").value="Adult";
    document.getElementById("report_channel").value="App";
    document.getElementById("behavior").value="Calm";
    document.getElementById("mobility").value="Normal";
    document.getElementById("traffic_risk").value="Low";
    document.getElementById("weather").value="Clear";
    document.getElementById("visible_injury").value="No";
    document.getElementById("animal_count").value=1;
    document.getElementById("near_traffic").value=0;
    document.getElementById("near_school").value=0;
    document.getElementById("very_thin").value=0;
    document.getElementById("bleeding").value=0;
    document.getElementById("open_wound").value=0;
    document.getElementById("pregnant_or_nursing").value=0;
    document.getElementById("has_collar").value=1;
    document.getElementById("reporter_confidence").value=3;
    document.getElementById("report_hour").value=14;
    document.getElementById("report_dayofweek").value=3;
    document.getElementById("report_month").value=5;
    predict();
  }
  function fillMed(){
    document.getElementById("neighborhood").value="Kombinat";
    document.getElementById("animal_type").value="Dog";
    document.getElementById("age_group").value="Adult";
    document.getElementById("report_channel").value="Call";
    document.getElementById("behavior").value="Fearful";
    document.getElementById("mobility").value="Limping";
    document.getElementById("traffic_risk").value="Medium";
    document.getElementById("weather").value="Rain";
    document.getElementById("visible_injury").value="No";
    document.getElementById("animal_count").value=1;
    document.getElementById("near_traffic").value=1;
    document.getElementById("near_school").value=0;
    document.getElementById("very_thin").value=1;
    document.getElementById("bleeding").value=0;
    document.getElementById("open_wound").value=0;
    document.getElementById("pregnant_or_nursing").value=0;
    document.getElementById("has_collar").value=0;
    document.getElementById("reporter_confidence").value=4;
    document.getElementById("report_hour").value=18;
    document.getElementById("report_dayofweek").value=2;
    document.getElementById("report_month").value=11;
    predict();
  }
  function fillHigh(){
    document.getElementById("neighborhood").value="Kombinat";
    document.getElementById("animal_type").value="Dog";
    document.getElementById("age_group").value="Senior";
    document.getElementById("report_channel").value="Call";
    document.getElementById("behavior").value="Weak";
    document.getElementById("mobility").value="Immobile";
    document.getElementById("traffic_risk").value="High";
    document.getElementById("weather").value="Heatwave";
    document.getElementById("visible_injury").value="Yes";
    document.getElementById("animal_count").value=1;
    document.getElementById("near_traffic").value=1;
    document.getElementById("near_school").value=1;
    document.getElementById("very_thin").value=1;
    document.getElementById("bleeding").value=1;
    document.getElementById("open_wound").value=1;
    document.getElementById("pregnant_or_nursing").value=0;
    document.getElementById("has_collar").value=0;
    document.getElementById("reporter_confidence").value=5;
    document.getElementById("report_hour").value=21;
    document.getElementById("report_dayofweek").value=5;
    document.getElementById("report_month").value=8;
    predict();
  }
</script>
</body>
</html>
"""
