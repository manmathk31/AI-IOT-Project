# AI-Driven Wind Turbine Monitoring System
### Team Tech Titans — Manmath & Vishal

---

## Project Overview

End-to-end intelligent wind turbine monitoring platform. IoT sensors feed real-time data through an ML inference pipeline that predicts Normal / Warning / Fault / Critical Flame states. A modern web dashboard gives operators live visibility, alert management, and maintenance tracking.

**Phase 1:** Full web app with simulated sensor data + trained ML model from Colab.
**Phase 2:** Swap simulator for real Arduino serial stream. Zero other changes.

---

## Team Division

| Member | Owns |
|--------|------|
| Manmath | Backend (FastAPI) + ML inference pipeline |
| Vishal | Frontend (HTML/CSS/JS) — complete dashboard |

---

## Project Structure

```
wind-turbine-monitor/
│
├── backend/
│   ├── app.py
│   ├── simulator.py
│   ├── inference_engine.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── database.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── predictions.py
│   │   ├── alerts.py
│   │   └── maintenance.py
│   ├── ml/
│   │   ├── model_v1.pkl          ← paste from Colab after training
│   │   ├── scaler.pkl            ← paste from Colab after training
│   │   └── metrics.json          ← paste from Colab after training
│   ├── requirements.txt
│   └── .env
│
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── style.css
│   └── js/
│       ├── dashboard.js
│       ├── ai_insights.js
│       ├── alerts.js
│       ├── maintenance.js
│       └── analytics.js
│
└── README.md
```

---

## API Contract (Both members must match this exactly)

```
GET  /api/live-data
     → { readings: [{ id, timestamp, temp, vibration, current, flame }] }

GET  /api/prediction
     → { prediction, confidence, override, timestamp }

GET  /api/history?limit=100
     → { readings: [{ id, timestamp, temp, vibration, current, flame }] }

GET  /api/alerts
     → { alerts: [{ id, type, severity, message, status, timestamp }] }

POST /api/alerts/{id}/resolve
     → { success: true }

GET  /api/maintenance
     → { tasks: [{ id, engineer, machine, notes, status, timestamp }] }

POST /api/maintenance
     body: { engineer, machine, notes }
     → { id, engineer, machine, notes, status, timestamp }

PATCH /api/maintenance/{id}
     body: { status }    "Pending" | "In Progress" | "Completed"
     → { id, status }

GET  /api/model-metrics
     → { accuracy, f1, precision, recall,
         confusion_matrix: [[],[],[],[]],
         feature_importances: { feature_name: value } }
```

**severity values:** "info" | "warning" | "fault" | "critical"
**prediction values:** "Normal" | "Warning" | "Fault" | "CRITICAL_FLAME"

---

## How to Run

```bash
# Terminal 1 — Backend
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# Terminal 2 — Frontend
# Just open frontend/index.html in browser
# Or serve with: python -m http.server 5500 (from frontend/ folder)
```

---
---

# MANMATH — BACKEND AGENT PROMPT

> Copy everything below this line until the divider and paste it to your Claude agent as one single message.

---

I need you to build a complete, production-quality FastAPI backend for an AI-driven wind turbine fault detection and predictive maintenance system. This is a serious project so write clean, well-structured, fully working code for every single file. Do not skip anything, do not write placeholder comments saying "add logic here". Every function must be fully implemented.

## Tech stack
- FastAPI
- SQLAlchemy with SQLite (file: turbine.db)
- joblib for loading ML models
- scikit-learn, numpy
- python-dotenv
- threading and queue from Python standard library
- uvicorn

## What this system does
A simulator generates fake sensor readings (temperature, current, vibration, flame) every second and pushes them into a Python queue. An inference engine runs in a separate thread, consumes from that queue, maintains a sliding window of 20 readings, extracts features, and runs an ML model to classify each window as Normal / Warning / Fault / CRITICAL_FLAME. Results are saved to SQLite. FastAPI serves this data through REST endpoints. A frontend dashboard (built separately) polls these endpoints every 2 seconds.

## Files to create

---

### backend/database.py

Set up SQLAlchemy with SQLite. Database file is `turbine.db` in the backend folder. Create engine with `connect_args={"check_same_thread": False}`. Create SessionLocal with autocommit=False, autoflush=False. Create Base = declarative_base(). Write a `get_db()` generator function that yields a session and closes it in a finally block. Export: engine, SessionLocal, Base, get_db.

---

### backend/models.py

Import Base from database. Create four SQLAlchemy ORM models:

**SensorReading:**
- id: Integer, primary key, autoincrement
- timestamp: DateTime, default=datetime.utcnow
- temp: Float (temperature in Celsius)
- vibration: Integer (0 or 1)
- current: Float (current in mA)
- flame: Integer (0 or 1)

**Prediction:**
- id: Integer, primary key, autoincrement
- timestamp: DateTime, default=datetime.utcnow
- prediction: String (Normal/Warning/Fault/CRITICAL_FLAME)
- confidence: Float (0.0 to 1.0)
- override: Boolean, default False

**Alert:**
- id: Integer, primary key, autoincrement
- timestamp: DateTime, default=datetime.utcnow
- type: String (e.g. "Fault Detected", "Flame Detected", "Warning")
- severity: String ("info", "warning", "fault", "critical")
- message: String (detailed message)
- status: String, default="active" ("active" or "resolved")

**MaintenanceTask:**
- id: Integer, primary key, autoincrement
- timestamp: DateTime, default=datetime.utcnow
- engineer: String
- machine: String
- notes: String
- status: String, default="Pending" ("Pending", "In Progress", "Completed")

---

### backend/feature_engineering.py

Write one function: `extract_features(window)`.

`window` is a list of exactly 20 dicts, each with keys: temp (float), vibration (int), current (float), flame (int).

Extract and return two things — a dict and a numpy array (in same fixed order):

Features to extract:
- temp_mean: mean of all temp values in window
- temp_std: standard deviation of all temp values
- temp_rate_of_change: last temp value minus first temp value in window
- temp_max: maximum temp in window
- current_mean: mean of all current values
- current_std: standard deviation of all current values
- current_spike: max current minus mean current (how spikey it got)
- current_rate_of_change: last current minus first current
- vibration_count: sum of all vibration values (how many readings had vibration=1)
- flame_count: sum of all flame values

Return a tuple: (feature_dict, numpy_array) where numpy_array contains these 10 values in this exact order: [temp_mean, temp_std, temp_rate_of_change, temp_max, current_mean, current_std, current_spike, current_rate_of_change, vibration_count, flame_count].

---

### backend/simulator.py

Write a function `run_simulator(data_queue)` that runs forever in an infinite loop with `time.sleep(1)` per iteration.

The simulator cycles through four operating states. Use a state machine with a counter tracking how many seconds have been spent in the current state:

**State 0 — Normal (runs for 30 seconds):**
- temp: random.gauss(42, 3), clipped to 30-55
- current: random.gauss(285, 15), clipped to 240-330
- vibration: 1 if random.random() < 0.05 else 0 (rare)
- flame: 0

**State 1 — Warning (runs for 15 seconds):**
- temp: random.gauss(62, 4), clipped to 52-72
- current: random.gauss(385, 20), clipped to 340-430
- vibration: 1 if random.random() < 0.35 else 0 (occasional)
- flame: 0

**State 2 — Fault (runs for 10 seconds):**
- temp: random.gauss(76, 3), clipped to 68-84
- current: random.gauss(472, 12), clipped to 445-500
- vibration: 1 if random.random() < 0.75 else 0 (frequent)
- flame: 0

**State 3 — Critical Flame (runs for 5 seconds):**
- temp: random.gauss(82, 2), clipped to 74-88
- current: random.gauss(490, 8), clipped to 470-500
- vibration: 1
- flame: 1

After each state completes its duration, move to the next state. After state 3, loop back to state 0. 

Each iteration: build a dict with keys timestamp (datetime.utcnow().isoformat()), temp, vibration, current, flame. Put it in data_queue. Log the current state and reading values to console so we can see it working.

---

### backend/inference_engine.py

Write a function `run_inference(data_queue, session_factory)`.

At the top of the function, before the loop:
- Try to load `ml/model_v1.pkl` and `ml/scaler.pkl` using joblib. If files do not exist, set both to None and print a warning saying "Model files not found — running in simulation mode. Predictions will be placeholder only."
- Set label_map = {0: "Normal", 1: "Warning", 2: "Fault", 3: "CRITICAL_FLAME"}
- Create a collections.deque with maxlen=20 as the sliding window

In the infinite loop:
1. Get next reading from data_queue (blocking get)
2. Append reading to the sliding window deque
3. If len(window) < 20: continue (wait until window is full)
4. Open a new DB session using session_factory
5. Save the raw reading as a new SensorReading to DB
6. Check if the latest reading has flame == 1:
   - If yes: create Prediction(prediction="CRITICAL_FLAME", confidence=1.0, override=True)
   - Create Alert(type="Flame Detected", severity="critical", message="Flame sensor triggered on turbine. Emergency shutdown recommended. Contact safety team immediately.")
   - Save both to DB, commit, continue to next iteration
7. If flame == 0 AND model is loaded:
   - Call extract_features(list(window)) to get (feat_dict, feat_array)
   - Scale feat_array using scaler.transform([feat_array])
   - Call model.predict_proba([scaled_array])
   - Get predicted class index = argmax of probabilities
   - Get confidence = max of probabilities
   - Map index to label using label_map
   - Save Prediction to DB
   - If label == "Warning": create Alert(type="Warning", severity="warning", message=f"System showing warning signs. Temp: {latest_reading['temp']:.1f}C, Current: {latest_reading['current']:.1f}mA. Monitor closely.")
   - If label == "Fault": create Alert(type="Fault Detected", severity="fault", message=f"Fault condition detected. Temp: {latest_reading['temp']:.1f}C, Current: {latest_reading['current']:.1f}mA, Vibration active. Schedule maintenance immediately.")
   - If label == "Normal": no alert
8. If flame == 0 AND model is NOT loaded:
   - Save Prediction(prediction="Simulating — No Model", confidence=0.0, override=False)
9. Commit session, close session. Catch all exceptions, rollback on error, log error.

---

### backend/routes/data.py

Create an APIRouter. All routes prefixed with /api.

**GET /api/live-data:**
Query last 50 SensorReading records ordered by timestamp descending. Return JSON:
```json
{
  "readings": [
    { "id": 1, "timestamp": "...", "temp": 42.3, "vibration": 0, "current": 285.2, "flame": 0 }
  ]
}
```
Convert each ORM object to dict manually (do not rely on Pydantic for this, just build the dict yourself).

**GET /api/history:**
Accept optional query parameter `limit` (int, default=100, max=500). Return last N SensorReading records ordered by timestamp ascending (so charts render left to right in time order). Same response format as live-data under key "readings".

---

### backend/routes/predictions.py

Create an APIRouter.

**GET /api/prediction:**
Query the single most recent Prediction record. Return:
```json
{
  "prediction": "Normal",
  "confidence": 0.94,
  "override": false,
  "timestamp": "..."
}
```
If no prediction exists yet return `{"prediction": "Initializing", "confidence": 0.0, "override": false, "timestamp": null}`.

**GET /api/model-metrics:**
Read the file `ml/metrics.json` from the backend folder. Return its contents directly as JSON. If the file does not exist, return this placeholder so the frontend never breaks:
```json
{
  "accuracy": 0.96,
  "f1": 0.95,
  "precision": 0.94,
  "recall": 0.95,
  "confusion_matrix": [
    [145, 3, 1, 0],
    [2, 88, 4, 0],
    [1, 3, 72, 1],
    [0, 0, 1, 38]
  ],
  "feature_importances": {
    "temp_mean": 0.21,
    "temp_std": 0.09,
    "temp_rate_of_change": 0.08,
    "temp_max": 0.14,
    "current_mean": 0.18,
    "current_std": 0.07,
    "current_spike": 0.10,
    "current_rate_of_change": 0.06,
    "vibration_count": 0.04,
    "flame_count": 0.03
  }
}
```

---

### backend/routes/alerts.py

Create an APIRouter.

**GET /api/alerts:**
Query all Alert records ordered by timestamp descending. Return:
```json
{
  "alerts": [
    { "id": 1, "type": "Fault Detected", "severity": "fault", "message": "...", "status": "active", "timestamp": "..." }
  ]
}
```

**POST /api/alerts/{id}/resolve:**
Find Alert by id. If not found raise 404. Set status = "resolved". Commit. Return `{"success": true}`.

---

### backend/routes/maintenance.py

Create an APIRouter.

**GET /api/maintenance:**
Query all MaintenanceTask records ordered by timestamp descending. Return under key "tasks".

**POST /api/maintenance:**
Accept JSON body with fields: engineer (str), machine (str), notes (str). Create new MaintenanceTask with status="Pending". Save and commit. Return the created task as JSON with all fields including id and timestamp.

**PATCH /api/maintenance/{id}:**
Accept JSON body with field: status (str). Find task by id, raise 404 if not found. Update status field. Commit. Return updated task as JSON.

---

### backend/app.py

Create the main FastAPI application. Do the following in order:

1. Instantiate `app = FastAPI(title="Wind Turbine Monitoring API", version="1.0.0")`

2. Add CORSMiddleware:
   - allow_origins=["*"]
   - allow_credentials=True
   - allow_methods=["*"]
   - allow_headers=["*"]

3. Import and include all four routers (data, predictions, alerts, maintenance). Do not add any extra prefix — the routes already include /api/ in their paths.

4. Add a startup event using @app.on_event("startup"):
   - Call Base.metadata.create_all(bind=engine) to create all tables
   - Create a queue.Queue() instance called sensor_queue
   - Start simulator: t1 = threading.Thread(target=run_simulator, args=(sensor_queue,), daemon=True); t1.start()
   - Start inference engine: t2 = threading.Thread(target=run_inference, args=(sensor_queue, SessionLocal), daemon=True); t2.start()
   - Print "System started. Simulator and inference engine running."

5. Add a root GET / route that returns `{"status": "Wind Turbine Monitoring API is running", "docs": "/docs"}`.

---

### backend/requirements.txt

```
fastapi
uvicorn[standard]
sqlalchemy
joblib
scikit-learn
numpy
python-dotenv
imbalanced-learn
```

---

### backend/ml/metrics.json

Create this file with realistic fake model metrics so the frontend works before the real Colab model is ready:

```json
{
  "accuracy": 0.962,
  "f1": 0.958,
  "precision": 0.961,
  "recall": 0.955,
  "confusion_matrix": [
    [145, 3, 1, 0],
    [2, 88, 4, 0],
    [1, 3, 72, 1],
    [0, 0, 1, 38]
  ],
  "class_labels": ["Normal", "Warning", "Fault", "Critical Flame"],
  "feature_importances": {
    "temp_mean": 0.21,
    "temp_std": 0.09,
    "temp_rate_of_change": 0.08,
    "temp_max": 0.14,
    "current_mean": 0.18,
    "current_std": 0.07,
    "current_spike": 0.10,
    "current_rate_of_change": 0.06,
    "vibration_count": 0.04,
    "flame_count": 0.03
  }
}
```

---

After building all files, show me how to run the backend with one command and confirm all endpoints work by listing them. The command to run is: `uvicorn app:app --reload --port 8000` from inside the backend/ folder.

---
---

# VISHAL — FRONTEND AGENT PROMPT

> Copy everything below this line and paste it to your Claude agent as one single message.

---

I need you to build a complete, world-class frontend dashboard for an AI-driven wind turbine monitoring system. This is a serious project with a demo in front of judges. The UI must be beautiful, modern, clean, and highly professional — the kind of interface that makes judges say "wow" the moment they see it.

## Design direction

Think clean industrial monitoring meets modern SaaS product design. NOT dark/cyberpunk. NOT generic bootstrap. Think the design quality of Linear, Vercel, or Notion — light background, excellent typography, purposeful color, generous whitespace, smooth micro-interactions. Every element should feel intentional and crafted.

**Color palette:**
- Background: #F8F9FC (very light blue-grey)
- Surface/cards: #FFFFFF with subtle shadow
- Primary accent: #2563EB (bold blue — used for active states, primary actions)
- Success/Normal: #16A34A (green)
- Warning: #D97706 (amber)
- Fault/Danger: #DC2626 (red)
- Critical Flame: #991B1B with a warm orange tint
- Text primary: #111827
- Text secondary: #6B7280
- Borders: #E5E7EB

**Typography:**
Use Google Fonts. Import `DM Sans` for all UI text (weights 400, 500, 600) and `Space Mono` for numeric values, sensor readings, and data values — the monospace font makes numbers look technical and precise.

**Feel:** Smooth. Every hover has a subtle transition (150ms ease). Cards lift slightly on hover. Buttons have satisfying press states. Status changes animate. The flame alert entrance is dramatic. Charts are clean with no unnecessary gridlines.

## Tech stack
- Pure HTML, CSS, JavaScript — no frameworks
- Chart.js from CDN for all charts
- Google Fonts via @import
- Backend API is at http://localhost:8000

## Project structure to create

```
frontend/
├── index.html
├── css/
│   └── style.css
└── js/
    ├── dashboard.js
    ├── ai_insights.js
    ├── alerts.js
    ├── maintenance.js
    └── analytics.js
```

---

## index.html

Build the full page structure. Include:

1. Google Fonts import in `<head>` for DM Sans (400, 500, 600) and Space Mono (400)
2. Chart.js CDN: `https://cdn.jsdelivr.net/npm/chart.js`
3. Link to css/style.css
4. All 5 JS files at bottom of body

**Page layout:**
- A fixed left sidebar, 240px wide, full height. Contains: app logo at top ("TurbineAI" with a small wind turbine SVG icon inline), then navigation links for all 5 tabs. Active tab has blue left border and blue text. Below nav links show a small "System Status" indicator — a green/red dot with text "Live" or "Offline" (id="system-status").
- A top header bar: shows current tab title on left, current date/time on right (updates every second), and a small avatar placeholder saying "Team Tech Titans" on far right.
- Main content area to the right of sidebar. Padding 32px.
- Flame alert banner: ABOVE the sidebar layout, fixed to top, full width, hidden by default. id="flame-banner". Deep red background (#7F1D1D), white text, padding 14px 24px. Text: "🔥 CRITICAL ALERT — Flame Detected on Turbine. Immediate action required. Contact safety team now." Center aligned. Has an X close button on the right. When shown, push all content down (not overlay). CSS animation: slide down from top when shown.

**5 content sections** with ids: section-dashboard, section-insights, section-alerts, section-maintenance, section-analytics. Only the active one is visible (display block vs none).

---

## css/style.css

Write comprehensive CSS. Key things to include:

**Reset and base:** *, box-sizing border-box, margin 0, padding 0. Body uses DM Sans, background #F8F9FC, color #111827.

**Sidebar:** Fixed left, 240px wide, full height, white background, border-right 1px solid #E5E7EB, padding 24px 0. Logo area: padding 0 24px 32px. Nav links: block elements, padding 12px 24px, no underline, color #6B7280, font-weight 500, font-size 14px, border-left 3px solid transparent. On hover: background #F3F4F6, color #111827. Active class: border-left 3px solid #2563EB, color #2563EB, background #EFF6FF.

**Main layout:** margin-left 240px. Header: white, border-bottom, padding 0 32px, height 64px, flex row, align-items center, justify-content space-between. Content area: padding 32px.

**Cards:** background white, border-radius 12px, border 1px solid #E5E7EB, padding 24px, box-shadow 0 1px 3px rgba(0,0,0,0.04). On hover: box-shadow 0 4px 12px rgba(0,0,0,0.08), transform translateY(-1px). Transition 150ms ease.

**Status badge:** Large pill badge for the prediction state. Padding 10px 24px, border-radius 9999px, font-weight 600, font-size 18px, display inline-flex, align-items center, gap 8px. Colors by class: .badge-normal (bg #DCFCE7, color #15803D), .badge-warning (bg #FEF3C7, color #92400E), .badge-fault (bg #FEE2E2, color #991B1B), .badge-critical (bg #7F1D1D, color white, animation pulse 1s infinite).

**LED indicators:** 12px circles. .led (bg #D1D5DB). .led-active-vibration (bg #F59E0B, box-shadow 0 0 8px #F59E0B). .led-active-flame (bg #DC2626, box-shadow 0 0 10px #DC2626, animation pulse 0.6s infinite).

**Metric cards:** Compact cards showing a single number. Icon area on left (48px circle with light colored background). Right side shows label (12px, secondary text) and value (24px, font Space Mono, primary text).

**Alert table:** full width, border-collapse collapse. Thead: font-size 12px, uppercase, letter-spacing 0.05em, color #6B7280, border-bottom 2px solid #E5E7EB. Tbody rows: border-bottom 1px solid #F3F4F6. Row severity tints: .row-info (background #EFF6FF), .row-warning (background #FFFBEB), .row-fault (background #FFF1F2), .row-critical (background #FFF1F2, border-left 3px solid #991B1B).

**Badges for severity:** Small pill badges. .sev-info (bg #DBEAFE, color #1D4ED8), .sev-warning (bg #FEF3C7, color #92400E), .sev-fault (bg #FEE2E2, color #991B1B), .sev-critical (bg #7F1D1D, color white).

**Kanban board:** Three columns side by side using CSS grid (grid-template-columns: 1fr 1fr 1fr, gap 20px). Column header: font-weight 600, margin-bottom 16px, padding-bottom 12px, border-bottom 2px solid respective color (Pending=#E5E7EB, In Progress=#DBEAFE, Completed=#DCFCE7). Cards: white, border 1px solid #E5E7EB, border-radius 8px, padding 16px, margin-bottom 12px, border-top 3px solid respective color.

**Form inputs:** width 100%, padding 10px 14px, border 1px solid #D1D5DB, border-radius 8px, font-family DM Sans, font-size 14px. On focus: outline none, border-color #2563EB, box-shadow 0 0 0 3px rgba(37,99,235,0.1).

**Buttons:** Primary: bg #2563EB, color white, padding 10px 20px, border-radius 8px, border none, font-weight 500, cursor pointer. On hover: bg #1D4ED8. On active: transform scale(0.98). Secondary: bg white, border 1px solid #D1D5DB, color #374151. Small buttons (table actions): padding 6px 12px, font-size 13px.

**Confusion matrix table:** All cells equal width, padding 16px, text-align center, font-family Space Mono. Header cells: bg #F9FAFB, font-weight 600, font-size 12px uppercase. Value cells: color-coded by intensity using inline style (set by JS).

**Animations:** @keyframes pulse (opacity 1 → 0.5 → 1). @keyframes slideDown (translateY(-100%) → translateY(0)). @keyframes fadeIn (opacity 0 → 1). Chart containers: height fixed at 280px so all charts are uniform.

---

## js/dashboard.js

This manages the Live Monitoring tab (section-dashboard).

**HTML structure to create for this tab (inject via JS or define in index.html):**
- Top row: 4 metric cards side by side. Card 1: "Temperature" — shows latest temp value in Space Mono with °C. Card 2: "Current" — shows latest current in Space Mono with mA. Card 3: "Vibration" — LED indicator + text "Active"/"Inactive". Card 4: "Flame" — LED indicator + text "Detected"/"Clear".
- Below that: a large status section showing the prediction badge (big, centered), confidence percentage below it, and timestamp of last prediction.
- Below that: two Chart.js charts side by side — Temperature over time (line chart, blue), Current over time (line chart, orange).

**JS logic:**

```javascript
const API = 'http://localhost:8000';
let tempChart, currentChart;
let tempData = [], currentData = [], timeLabels = [];

function initDashboard() {
  // Initialize Chart.js charts
  // tempChart: line chart on canvas#temp-chart
  // Blue color: #2563EB, no fill, tension 0.4, no point radius, smooth
  // currentChart: line chart on canvas#current-chart  
  // Orange color: #D97706, same styling
  // Both charts: no legend, minimal grid (only horizontal lines in #F3F4F6), 
  //   x-axis shows time labels (HH:MM:SS format), max 20 data points shown
  // Start polling
  pollLiveData();
  pollPrediction();
  setInterval(pollLiveData, 2000);
  setInterval(pollPrediction, 2000);
}

async function pollLiveData() {
  // fetch /api/live-data
  // Take latest reading from readings[0]
  // Update temp metric card value
  // Update current metric card value
  // Update vibration LED and text
  // Update flame LED and text
  // Add latest reading to chart data arrays (keep max 20 points)
  // Update both charts
}

async function pollPrediction() {
  // fetch /api/prediction
  // Update status badge: remove all badge-* classes, add correct one
  // Update badge text to prediction value (replace underscores with spaces)
  // Update confidence text as percentage
  // If prediction is CRITICAL_FLAME: show flame banner (remove hidden class from #flame-banner)
  // Else: hide flame banner
}

export { initDashboard };
```

Write the full implementation — not just the structure above. All chart initialization, all DOM updates, all fetch calls, all error handling.

---

## js/ai_insights.js

This manages the AI Insights tab (section-insights).

**HTML structure for this tab:**
- Top row: 4 metric cards showing model performance: Accuracy, F1 Score, Precision, Recall — all as percentages in Space Mono font.
- Below: two cards side by side. Left card: Confusion Matrix. Right card: Feature Importances chart.
- Below: a card showing current prediction with large confidence bar.

**JS logic:**

```javascript
async function initInsights() {
  const data = await fetch('http://localhost:8000/api/model-metrics').then(r => r.json());
  
  // Update 4 metric cards with accuracy, f1, precision, recall as percentages
  
  // Render confusion matrix as HTML table
  // 4x4 table, row headers = Actual [Normal, Warning, Fault, C.Flame]
  // column headers = Predicted same labels
  // Each value cell: find max value in matrix, compute intensity = value/max
  // Background color: rgba(37, 99, 235, intensity * 0.7) — blue intensity
  // Text color: intensity > 0.5 ? white : #111827
  // Values in Space Mono font
  
  // Render feature importances as horizontal Chart.js bar chart
  // Sort features by importance descending
  // Blue bars (#2563EB), horizontal orientation
  // Clean labels: replace underscores with spaces, title case
  // Show values as labels on bars
  
  // Also fetch latest prediction and show with animated confidence bar
}
```

Write the full implementation.

---

## js/alerts.js

This manages the Alerts tab (section-alerts).

**HTML structure for this tab:**
- A summary row of 3 cards: "Total Alerts" (count), "Active Alerts" (count of status=active), "Critical Alerts" (count of severity=critical).
- A filter row: buttons for All / Active / Resolved — filter the table client-side.
- Full-width table with columns: Time, Type, Severity (badge), Message, Status, Action.

**JS logic:**

```javascript
let allAlerts = [];

async function initAlerts() {
  await fetchAlerts();
  setInterval(fetchAlerts, 5000);
}

async function fetchAlerts() {
  // fetch /api/alerts
  // Store in allAlerts
  // Update 3 summary cards
  // Render table
}

function renderAlertsTable(filter = 'all') {
  // Filter allAlerts based on filter param
  // Render table rows
  // Each row has severity-appropriate row class and tint
  // Severity shown as colored pill badge
  // If status is active: show blue "Resolve" button
  //   onClick: call POST /api/alerts/{id}/resolve then fetchAlerts()
  // If resolved: show grey "Resolved" text
  // Time formatted as: "14:23:05 — Jan 15, 2025"
}
```

Write full implementation.

---

## js/maintenance.js

This manages the Maintenance tab (section-maintenance).

**HTML structure for this tab:**
- A form card at top: "Assign Maintenance Task". Fields: Engineer Name (text), Machine (text, placeholder "e.g. Turbine Unit 1"), Notes (textarea, 3 rows), Submit button "Assign Task". Form has nice spacing, labels above each input.
- Below: Kanban board with 3 columns: Pending (grey accent), In Progress (blue accent), Completed (green accent). Column headers show count of cards in that column.

**JS logic:**

```javascript
let allTasks = [];

async function initMaintenance() {
  await fetchTasks();
}

async function fetchTasks() {
  // fetch /api/maintenance
  // Store in allTasks
  // Render kanban board
}

function renderKanban() {
  // Separate tasks into 3 arrays by status
  // Update column header counts
  // Render task cards in each column
  // Each card shows: engineer name (bold), machine name, notes (truncated to 80 chars), 
  //   formatted timestamp, action button
  // Pending cards: blue "Start Work" button → PATCH to "In Progress"
  // In Progress cards: green "Mark Complete" button → PATCH to "Completed"  
  // Completed cards: green checkmark icon, no button, slightly muted opacity
}

async function submitTask(engineer, machine, notes) {
  // POST /api/maintenance with body
  // On success: clear form fields, call fetchTasks
}

async function updateStatus(id, status) {
  // PATCH /api/maintenance/{id} with { status }
  // On success: fetchTasks
}
```

Attach form submit event listener. Write full implementation.

---

## js/analytics.js

This manages the Analytics tab (section-analytics).

**HTML structure for this tab:**
- Top row: 4 summary stat cards: Max Temperature recorded, Average Current, Total Readings, Total Faults detected.
- Below: 2x2 grid of charts. Top-left: Temperature trend (line). Top-right: Current trend (line). Bottom-left: Prediction distribution (doughnut chart — 4 slices for Normal/Warning/Fault/CriticalFlame with colors green/amber/red/dark red). Bottom-right: Alerts by severity (horizontal bar chart).

**JS logic:**

```javascript
async function initAnalytics() {
  const historyRes = await fetch('http://localhost:8000/api/history?limit=200').then(r => r.json());
  const alertsRes = await fetch('http://localhost:8000/api/alerts').then(r => r.json());
  
  const readings = historyRes.readings;
  const alerts = alertsRes.alerts;
  
  // Compute summary stats and update 4 stat cards
  
  // Render temperature trend line chart (readings over time, x = timestamp, y = temp)
  
  // Render current trend line chart
  
  // Count prediction-like distribution from readings:
  //   Approximate by bucketing temp values:
  //   temp < 55 = Normal, 55-68 = Warning, 68-80 = Fault, >80 = Critical
  //   (This approximates distribution since we don't have per-reading predictions)
  // Render doughnut chart with these 4 buckets
  
  // Count alerts by severity and render horizontal bar chart
}
```

Write full implementation.

---

## Tab navigation (in index.html or a small inline script)

```javascript
const tabs = ['dashboard', 'insights', 'alerts', 'maintenance', 'analytics'];
const initFns = {
  dashboard: initDashboard,
  insights: initInsights,
  alerts: initAlerts,
  maintenance: initMaintenance,
  analytics: initAnalytics
};

function switchTab(tabName) {
  // Hide all sections
  // Show section-{tabName}
  // Update nav link active states
  // Update header title
  // Call corresponding init function
}

// On page load: show dashboard by default
switchTab('dashboard');

// Update clock every second
setInterval(() => {
  document.getElementById('header-time').textContent = new Date().toLocaleString();
}, 1000);

// Flame banner close button
document.getElementById('flame-banner-close').onclick = () => {
  document.getElementById('flame-banner').classList.add('hidden');
};
```

---

## Final requirements

1. Every fetch call must have try/catch. On error: log to console, do not crash the UI.
2. On page load, show a subtle loading state on charts before data arrives (grey placeholder).
3. All number values use Space Mono font.
4. Charts use consistent color scheme — temperature always blue (#2563EB), current always orange (#D97706), never random colors.
5. The app must look completely professional on a 1366x768 laptop screen (standard laptop). Test your layout mentally at this resolution.
6. Smooth transitions everywhere: tabs fade in (opacity + translateY animation 200ms). Cards animate in on tab switch with staggered delay.
7. The flame banner must be visually alarming — it should make a demo viewer feel urgency.
8. Empty states: if there are no alerts, show a clean empty state card saying "No alerts — system operating normally" with a green checkmark. Same for maintenance.
9. The sidebar "Live" status dot should turn green/pulse when data is being received and red if the last API call failed.

Build all 6 files completely. No placeholder code, no "// TODO" comments. Every function fully implemented.

---
---

## Integration Checklist

When both are done, run backend on port 8000, open frontend/index.html in browser. These must all work:

- [ ] Charts populate within 2 seconds of page load
- [ ] Status badge cycles through Normal → Warning → Fault → Critical Flame as simulator runs
- [ ] Flame banner appears and disappears with flame state
- [ ] LEDs light up correctly for vibration and flame
- [ ] Alerts table auto-refreshes and resolve button works
- [ ] Maintenance form submits and cards appear in correct Kanban column
- [ ] Move to In Progress and Complete buttons work
- [ ] AI Insights loads confusion matrix and feature importance chart
- [ ] Analytics charts populate with historical data
- [ ] No CORS errors in browser console (F12 → Console)
- [ ] No 404 errors on any endpoint

**If CORS errors occur:** Manmath check that CORSMiddleware has allow_origins=["*"]
**If data doesn't load:** Vishal check that API base URL is exactly `http://localhost:8000`
**If field names don't match:** Both check the API contract table in this README

---

## Phase 2 — Arduino Integration (later)

Manmath: replace `simulator.py` with `serial_listener.py`. The new file reads from the Arduino serial port and pushes dicts with the same keys (timestamp, temp, vibration, current, flame) into the same queue. The rest of the system — inference engine, routes, frontend — stays completely unchanged.

Vishal: zero changes needed.
