# Traffic Optimization System Using Multi-Modal AI

## Overview

This project presents an intelligent traffic optimization system that integrates multiple artificial intelligence models to analyze real-time traffic conditions, predict future congestion, detect anomalies, understand contextual events, and optimize traffic signal control.

The system follows a modular and interpretable architecture suitable for smart city applications.

---

## System Architecture

The Traffic Optimization System is composed of five independent yet interconnected AI models. Each model performs a specific function within a structured sensing–understanding–prediction–decision pipeline.

---

## Model 1: Traffic Perception (Computer Vision)

A YOLOv8-based object detection model analyzes road images to detect traffic entities such as:

- Cars  
- Buses  
- Trucks  
- Motorcycles  
- Pedestrians  

Detected objects are counted and transformed into a traffic density score, providing a compact numerical representation of congestion levels.

---

## Model 2: Traffic Forecasting (Time-Series Prediction)

An LSTM model processes historical traffic density values to predict future congestion trends.

This enables proactive traffic management by estimating upcoming traffic conditions instead of reacting only to current states.

---

## Model 3: Anomaly Detection

A Variational Autoencoder (VAE) learns normal traffic behavior patterns and detects abnormal conditions such as:

- Sudden congestion spikes  
- Unusual traffic drops  
- Potential accident-related disturbances  

Anomalies are identified using reconstruction error thresholds applied to normalized time-series data.

---

## Model 4: Traffic Event Understanding (Natural Language Processing)

A BERT-based NLP model analyzes textual data such as:

- Traffic reports  
- Social media updates  
- Incident descriptions  

The model classifies traffic-related events including accidents, roadwork, weather disruptions, and normal traffic conditions.

---

## Model 5: Traffic Signal Control (Reinforcement Learning)

A reinforcement learning agent acts as the system's decision-making component.

It receives structured inputs from all previous models and learns optimal traffic signal control strategies through reward-based optimization, aiming to:

- Reduce congestion  
- Minimize waiting time  
- Improve traffic flow efficiency  

---

## Data Flow Pipeline

1. Camera images are processed by the perception model to compute traffic density.
2. Density values are stored as time-series data.
3. Forecasting and anomaly detection models analyze traffic behavior patterns.
4. Text-based traffic events are extracted using NLP.
5. All outputs are combined into a unified state representation.
6. The reinforcement learning agent selects optimal traffic signal actions.

---

## Technologies Used

- Python  
- PyTorch  
- YOLOv8  
- LSTM  
- Variational Autoencoder (VAE)  
- BERT  
- Reinforcement Learning  

---

## Project Structure

```
Smart-Traffic-Optimization-System/
│
├── perception.py
├── prediction/
├── nlp/
├── rl/
├── data/        # Datasets (excluded from version control)
├── runs/        # Output visualizations
└── requirements.txt
```

---

## Installation

Clone the repository:

```
git clone https://github.com/hmusman2804045-max/Smart-Traffic-Optimization-System.git
cd Smart-Traffic-Optimization-System
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

Run the perception module:

```
python perception.py
```

Ensure required datasets are placed inside the `data/` directory before execution.

---

## Notes

- Large datasets and trained model weights are excluded from the repository.
- Synthetic data may be used for experimentation and demonstration.
- The system is designed for modular expansion and real-time smart city integration.

---

## License

This project is licensed under the MIT License.
