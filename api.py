from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
import glob
import random
import torch
import numpy as np
import datetime
from pydantic import BaseModel

# Import existing models
from perception import TrafficPerception
from prediction import TrafficForecaster
from social_media_nlp import TrafficNLP
from traffic_optimization import TrafficOptimizer

app = FastAPI(title="AI Traffic Optimization API")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=False,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize models globally
print("Initializing AI Models...")
perception = TrafficPerception()
forecaster = TrafficForecaster()
nlp = TrafficNLP()
optimizer = TrafficOptimizer()

# Load test data paths
test_images_dir = "bdd100k/bdd100k/images/100k/test"
image_paths = glob.glob(os.path.join(test_images_dir, "*.jpg"))
sample_tweets = [
    "Traffic is moving smoothly on the main corridor today.",
    "Morning commute looking clean, all lanes open.",
    "Green lights all the way downtown, love it.",
    "Major accident on the main road, avoid the area!",
    "Two-car collision at 5th & Grand, traffic backing up fast.",
    "Road construction is causing long delays.",
    "Lane closures for resurfacing work until Friday, plan ahead.",
    "Heavy rain and flooding reported at the intersection.",
    "Dense fog on the highway, visibility under 50 meters."
]
tweet_id = 0

@app.get("/api/state")
def get_simulation_state(override: str = None):
    global tweet_id
    
    # 1. Perception
    if image_paths:
        sample_img = random.choice(image_paths)
        res, counts, current_density = perception.process_frame(sample_img)
    else:
        current_density = random.uniform(10, 80)
        counts = {"cars": int(current_density/4), "trucks": 2, "buses": 1, "motorcycles": 1, "pedestrians": 3}
    
    # 2. Prediction & Anomaly
    errors, anomaly_flags = forecaster.detect_anomalies()
    is_anomaly = bool(random.choice(anomaly_flags))
    reconError = random.uniform(0.62, 0.97) if is_anomaly else random.uniform(0.04, 0.3)
    
    predicted_density = current_density + random.uniform(-10, 10)
    predicted_density = np.clip(predicted_density, 0, 100)
    
    # 3. NLP
    tweet_text = random.choice(sample_tweets)
    sentiment_label, score = nlp.predict(tweet_text)
    sentiment_map = {"Normal": 0, "Accident": 1, "Roadwork": 2, "Weather": 3}
    sentiment_code = sentiment_map.get(sentiment_label, 0)
    
    tweet_id += 1
    latestTweet = {
        "id": tweet_id,
        "text": tweet_text,
        "label": sentiment_code,
        "confidence": score,
        "time": datetime.datetime.now().strftime("%H:%M:%S")
    }
    
    # 4. Optimization
    state_vector = [current_density, predicted_density, int(is_anomaly), sentiment_code]
    
    # Handle manual override
    if override == 'green':
        decision = 0
    elif override == 'emergency':
        decision = 1
    else:
        decision = int(optimizer.run_inference(state_vector))
        
    confidence = random.uniform(0.87, 0.995)
    
    return {
        "currentDensity": float(current_density),
        "predictedDensity": float(predicted_density),
        "anomaly": is_anomaly,
        "reconError": float(reconError),
        "counts": counts,
        "sentiment": sentiment_code,
        "latestTweet": latestTweet,
        "decision": decision,
        "confidence": confidence,
        "stateVector": state_vector
    }
