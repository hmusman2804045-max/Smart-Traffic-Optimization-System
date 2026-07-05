# NEXUS Traffic Core — Frontend

A cyberpunk control-room dashboard for the multimodal AI Traffic Optimization System.
Built with **React 18 + Vite**, **Framer Motion**, and vanilla CSS design tokens
(glassmorphism, neon glows, animated SVG telemetry).

## Run

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

## Structure

```
src/
├── App.jsx                     # 3-column control-room layout
├── index.css                   # neon/glass design system (CSS variables)
├── hooks/
│   └── useSimulation.js        # front-end twin of main.py's simulation loop
└── components/
    ├── Header.jsx              # glowing brand + live system status chip
    ├── ControlDeck.jsx         # Start / Pause / Step / Reset + manual overrides
    ├── VisionPanel.jsx         # mock YOLOv8 feed with animated bounding boxes
    ├── DensityChart.jsx        # glowing SVG chart: current vs LSTM-predicted density
    ├── AnomalyAlert.jsx        # pulsing VAE anomaly banner
    ├── NLPFeed.jsx             # BERT-classified tweet stream with sentiment badges
    ├── DecisionCore.jsx        # 3D traffic light driven by the PPO action
    └── PipelineStatus.jsx      # per-model status (YOLOv8 / LSTM / VAE / BERT / PPO)
```

## Backend contract

`useSimulation.js` mirrors `main.py` exactly:

| Stage      | Backend module            | State produced                          |
| ---------- | ------------------------- | --------------------------------------- |
| Perception | `perception.py` (YOLOv8)  | vehicle counts, `current_density` 0–100 |
| Prediction | `prediction.py` (LSTM)    | `predicted_density` 0–100                |
| Anomaly    | `prediction.py` (VAE)     | reconstruction error, anomaly flag 0/1   |
| NLP        | `social_media_nlp.py`     | sentiment code 0–3 (Normal/Accident/Roadwork/Weather) |
| Optimizer  | `traffic_optimization.py` | PPO action: 0 = green/normal, 1 = emergency clearance |

To go live, replace `computeStep()` in `useSimulation.js` with a `fetch()` to an
endpoint that runs one step of `main.py` and returns
`{ currentDensity, predictedDensity, anomaly, counts, tweet, decision }`.
