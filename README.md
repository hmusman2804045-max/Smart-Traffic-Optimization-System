# Traffic Optimization System Using Multi-Modal AI

This project presents an intelligent traffic optimization system that integrates multiple artificial intelligence models to analyze real-time traffic conditions, predict future congestion, detect anomalies, understand contextual events, and optimize traffic signal control. The system follows a modular and interpretable design suitable for smart city applications.

## System Overview
The Traffic Optimization System is composed of five independent yet interconnected AI models. Each model is responsible for a specific task, allowing the system to sense, understand, predict, and act on traffic conditions in a structured pipeline.

### Model 1: Traffic Perception (Computer Vision)
A YOLOv8-based object detection model is used to analyze road images and detect traffic entities such as cars, buses, trucks, motorcycles, and pedestrians. The detected objects are counted and converted into a traffic density score, providing a compact numerical representation of road congestion.

### Model 2: Traffic Forecasting (Time-Series Prediction)
An LSTM model processes historical traffic density values to predict future congestion trends. This enables proactive traffic management by estimating upcoming traffic conditions rather than reacting only to the present state.

### Model 3: Anomaly Detection
A Variational Autoencoder (VAE) is used to learn normal traffic patterns and identify abnormal behavior such as sudden congestion spikes, drops, or potential accidents. Anomalies are detected using reconstruction error thresholds on normalized time-series data.

### Model 4: Traffic Event Understanding (Natural Language Processing)
A BERT-based NLP model analyzes text data such as traffic reports or social media messages to identify traffic-related events including accidents, roadwork, weather conditions, or normal situations. This model provides contextual information that cannot be obtained from cameras alone.

### Model 5: Traffic Signal Control (Reinforcement Learning)
A reinforcement learning agent acts as the decision-making component of the system. It receives structured inputs from all previous models and learns optimal traffic signal control strategies through rewards and penalties, aiming to reduce congestion and waiting time.

## Data Flow
1. Camera images are processed by the vision model to compute traffic density.
2. Density values are stored as time-series data.
3. Forecasting and anomaly detection models analyze traffic behavior.
4. Text-based traffic events are extracted using NLP.
5. All outputs are combined into a single state representation.
6. The reinforcement learning agent selects optimal traffic signal actions.

## Technologies Used
- Python
- Computer Vision (YOLOv8)
- Deep Learning (PyTorch)
- Time-Series Modeling (LSTM)
- Anomaly Detection (VAE)
- Natural Language Processing (BERT)
- Reinforcement Learning

## Project Structure
- `perception.py` – traffic detection and density calculation  
- `prediction/` – LSTM forecasting and VAE anomaly detection  
- `nlp/` – BERT-based text understanding  
- `rl/` – reinforcement learning traffic control  
- `data/` – datasets (excluded from version control)  
- `runs/` – inference and visualization outputs  

## Notes
- Large datasets and trained model weights are excluded from the repository.
- Synthetic data is used for demonstration and experimentation.
- The system is designed for modular expansion and real-time integration.

## License
This project is licensed under the MIT License.
