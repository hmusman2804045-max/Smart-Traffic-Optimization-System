# Smart Traffic Optimization System

## Overview

Smart Traffic Optimization System is an AI-driven framework designed for intelligent urban traffic management. The project integrates computer vision, time-series forecasting, anomaly detection, natural language processing, and reinforcement learning to enable adaptive and data-driven traffic control.

The objective is to reduce congestion, improve traffic efficiency, and support smart city infrastructure planning through advanced machine learning techniques.

---

## Features

- Vehicle detection and traffic density estimation using YOLOv8
- Traffic density prediction using LSTM networks
- Anomaly detection using Variational Autoencoders (VAE)
- Event classification using BERT
- Adaptive traffic signal control using Reinforcement Learning
- Visualization of prediction and anomaly outputs

---

## Project Structure

smart-traffic-optimization/
│
├── models/              # Trained and saved models
├── data/                # Dataset directory (not included)
├── docs/                # Output graphs and visualizations
├── src/                 # Source code files
├── requirements.txt     # Project dependencies
└── main.py              # Main execution file

---

## Sample Outputs

Sample visual outputs are available inside the `docs/` directory:

- Traffic density prediction graph
- Model output comparison
- Anomaly detection visualization

Example file paths:

docs/output_prediction_test.png  
docs/output_prediction_test.jpg  
docs/output_anomaly_test.png  

---

## Dataset

Datasets are not included in this repository due to size limitations.

To run the project locally:

1. Create a `data/` folder in the root directory.
2. Place the required dataset files inside the `data/` folder.

---

## Installation

Clone the repository:

git clone https://github.com/your-username/smart-traffic-optimization.git  
cd smart-traffic-optimization  

Install dependencies:

pip install -r requirements.txt  

---

## Usage

Run the main script:

python main.py  

Make sure required datasets are placed in the `data/` directory before execution.

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

## License

This project is licensed under the MIT License.
