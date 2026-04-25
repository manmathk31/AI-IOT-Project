# AI-Driven Wind Turbine Monitoring System  


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
│   │   ├── model_v1.pkl         
│   │   ├── scaler.pkl          
│   │   └── metrics.json      
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



Vishal: zero changes needed.
