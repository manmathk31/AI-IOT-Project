<div align="center">

  <img src="assets/wind-turbine.svg" alt="Green Wind Turbine" width="150" />

  <h1>🌪️ AI-Driven Wind Turbine Monitoring</h1>

  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&pause=1000&color=2ecc71&center=true&vCenter=true&width=600&lines=Intelligent+IoT+Platform;Real-time+ML+Diagnostics;Predictive+Maintenance+System" alt="Typing SVG" />

  <p>
    <b>Transforming raw sensor data into actionable intelligence with Machine Learning.</b>
  </p>

  <p>
    <a href="#-key-features"><img src="https://img.shields.io/badge/Features-Explore-brightgreen?style=for-the-badge&logo=appveyor" alt="Features" /></a>
    <a href="#-architecture--phases"><img src="https://img.shields.io/badge/Architecture-View-blue?style=for-the-badge&logo=cachet" alt="Architecture" /></a>
    <a href="#-getting-started"><img src="https://img.shields.io/badge/Get%20Started-Launch-orange?style=for-the-badge&logo=rocket" alt="Getting Started" /></a>
  </p>
  
  <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" alt="divider">
</div>

## ✨ The Vision

The **AI-Driven Wind Turbine Monitoring System** captures live IoT telemetry, funnels it through a state-of-the-art Machine Learning inference pipeline, and visualizes turbine health in a gorgeous web dashboard. 

<div align="center">
  <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Laptop.png" width="80" alt="Laptop" />
</div>

### 🎯 Predictive States
Our ML model accurately categorizes turbine health into four distinct states:
*   🟢 **Normal:** Optimal operation.
*   🟡 **Warning:** Anomalies detected, scheduled inspection advised.
*   🟠 **Fault:** Immediate attention required to prevent damage.
*   🔴 **Critical Flame:** Emergency shutdown protocol initiated!

---

## 🚀 Key Features

| Feature | Description |
| :--- | :--- |
| ⚡ **Real-Time Telemetry** | Live IoT sensor data streaming via serial or simulated endpoints. |
| 🧠 **Intelligent ML Inference** | Live predictions powered by a trained Scikit-Learn/Joblib model pipeline. |
| 📊 **Modern Dashboard** | A sleek, responsive HTML/JS/CSS frontend with dynamic visualizations. |
| 🚨 **Alert Management** | Automated alert generation and triaging based on predictive states. |
| 🛠️ **Maintenance Tracking** | Integrated task management for engineering teams. |

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" alt="divider">

## 🏗️ Architecture & Phases

<div align="center">
  <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Objects/Gear.png" width="80" alt="Gear" />
</div>

The project is structured into two seamless phases:

1.  **Phase 1 (Simulation):** Full web application driven by a synthetic sensor data simulator paired with our trained ML models.
2.  **Phase 2 (Hardware Integration):** Hot-swapping the simulator for a live Arduino serial stream—zero code changes required to the core pipeline.

### 💻 Tech Stack

<div align="center">
  <img src="https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi" alt="FastAPI" />
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python" />
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <br/>
  <img src="https://img.shields.io/badge/html5-%23E34F26.svg?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5" />
  <img src="https://img.shields.io/badge/css3-%231572B6.svg?style=for-the-badge&logo=css3&logoColor=white" alt="CSS3" />
  <img src="https://img.shields.io/badge/javascript-%23323330.svg?style=for-the-badge&logo=javascript&logoColor=%23F7DF1E" alt="JavaScript" />
</div>

---

## 📁 Repository Structure

```graphql
wind-turbine-monitor/
├── backend/                  # FastAPI & Core System
│   ├── app.py                # Main application entry point
│   ├── simulator.py          # Synthetic data generator
│   ├── serial_listener.py    # Arduino hardware integration
│   ├── inference_engine.py   # ML prediction pipeline
│   ├── feature_engineering.py# Data transformation logic
│   ├── data_collector_service.py # Telemetry aggregation
│   ├── routes/               # API Endpoints
│   ├── static/               # UI/UX Dashboard (HTML/CSS/JS)
│   └── ml/                   # Trained Models & Scalers
├── colab_training.py         # Google Colab model training script
├── data_collector.py         # Standalone data collection
└── README.md
```

<img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png" width="100%" alt="divider">

## 👥 Team Division

| Member | Responsibilities |
| :--- | :--- |
| **Manmath** | 🛠️ Backend Architecture (FastAPI), Database, ML Inference Pipeline |
| **Vishal** | 🎨 Frontend Design, Dashboard UI/UX, Data Visualization |

<div align="center">
  <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Hand%20gestures/Handshake.png" width="80" alt="Handshake" />
</div>

---

## ⚡ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/manmathk31/AI-IOT-Project.git
cd AI-IOT-Project
```

### 2. Setup the Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r backend/requirements.txt
```

### 3. Run the Application
```bash
python backend/app.py
```
*The API will be live at `http://localhost:8000`*

### 4. Open the Dashboard
The frontend UI is served directly via FastAPI. Visit `http://localhost:8000` in your web browser to view the dashboard!

<br>

<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=100&section=footer" width="100%" alt="Footer Wave" />
</div>
