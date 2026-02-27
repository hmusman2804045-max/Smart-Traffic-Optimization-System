import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(TrafficLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TrafficVAE(nn.Module):
    def __init__(self, input_dim=1, latent_dim=2):
        super(TrafficVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar

class TrafficForecaster:
    def __init__(self, data_path="traffic_data.csv"):
        print(" Initializing Traffic Forecaster...")
        self.data_path = data_path
        self.scaler = MinMaxScaler()
        self.lstm_model = TrafficLSTM()
        self.vae_model = TrafficVAE()
        self.lstm_path = "traffic_lstm.pth"
        self.vae_path = "traffic_vae.pth"
        if os.path.exists(self.lstm_path):
            print(" Loading saved LSTM weights...")
            self.lstm_model.load_state_dict(torch.load(self.lstm_path))
        if os.path.exists(self.vae_path):
            print(" Loading saved VAE weights...")
            self.vae_model.load_state_dict(torch.load(self.vae_path))
        df = pd.read_csv(data_path)
        self.raw_data = df['vehicle_count'].values.reshape(-1, 1)
        self.scaled_data = self.scaler.fit_transform(self.raw_data)

    def prepare_lstm_data(self, sequence_length=10):
        X, y = [], []
        for i in range(len(self.scaled_data) - sequence_length):
            X.append(self.scaled_data[i : i + sequence_length])
            y.append(self.scaled_data[i + sequence_length])
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

    def train_models(self, epochs=20):
        print("\n Training LSTM for Prediction...")
        optimizer_lstm = torch.optim.Adam(self.lstm_model.parameters(), lr=0.01)
        criterion_lstm = nn.MSELoss()
        X_lstm, y_lstm = self.prepare_lstm_data()
        for epoch in range(epochs):
            self.lstm_model.train()
            optimizer_lstm.zero_grad()
            outputs = self.lstm_model(X_lstm)
            loss = criterion_lstm(outputs, y_lstm)
            loss.backward()
            optimizer_lstm.step()
            if (epoch+1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        print("\n Training VAE for Anomaly Detection...")
        optimizer_vae = torch.optim.Adam(self.vae_model.parameters(), lr=0.01)
        criterion_vae = nn.MSELoss()
        data_vae = torch.FloatTensor(self.scaled_data)
        for epoch in range(epochs):
            self.vae_model.train()
            optimizer_vae.zero_grad()
            recon, mu, logvar = self.vae_model(data_vae)
            loss = criterion_vae(recon, data_vae)
            loss.backward()
            optimizer_vae.step()
            if (epoch+1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        print(" Training complete!")
        torch.save(self.lstm_model.state_dict(), self.lstm_path)
        torch.save(self.vae_model.state_dict(), self.vae_path)
        print(f" LSTM and VAE models saved to disk.")

    def detect_anomalies(self, threshold=0.05):
        self.vae_model.eval()
        data_tensor = torch.FloatTensor(self.scaled_data)
        with torch.no_grad():
            recon, _, _ = self.vae_model(data_tensor)
            errors = torch.abs(recon - data_tensor).numpy()
        anomalies = errors > threshold
        return errors, anomalies

if __name__ == "__main__":
    forecaster = TrafficForecaster()
    if hasattr(forecaster, 'scaled_data'):
        forecaster.train_models(epochs=30)
        errors, anomaly_flags = forecaster.detect_anomalies()
        print("\n Model 2 Summary:")
        anomaly_indices = np.where(anomaly_flags == True)[0]
        print(f" Number of Anomalies detected: {len(anomaly_indices)}")
        if len(anomaly_indices) > 0:
            print(f" First 3 Anomalies found at record indices: {anomaly_indices[:3]}")
        plt.figure(figsize=(12, 5))
        plt.plot(forecaster.raw_data, label="Actual Traffic", alpha=0.5)
        plt.scatter(anomaly_indices, forecaster.raw_data[anomaly_indices], color='red', label="Anomalies")
        plt.title("Traffic Flow & Detected Anomalies")
        plt.legend()
        plt.savefig("output_prediction_test.png")
        print("\n Result plot saved to output_prediction_test.png")
