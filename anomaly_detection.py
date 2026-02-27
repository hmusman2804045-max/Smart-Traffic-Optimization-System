import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

class VAE(nn.Module):
    def __init__(self, input_dim=1, latent_dim=1):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld_loss

def run_anomaly_detection(csv_file="traffic_data.csv"):
    if not os.path.exists(csv_file):
        print(f" Error: {csv_file} not found. Please run data_generator.py first.")
        return
    df = pd.read_csv(csv_file)
    data = df['vehicle_count'].values.reshape(-1, 1).astype(np.float32)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    data_tensor = torch.FloatTensor(data_scaled)
    mean = np.mean(data_scaled)
    std = np.std(data_scaled)
    normal_data = data_tensor[(data_tensor >= mean - 1.5*std) & (data_tensor <= mean + 1.5*std)].reshape(-1, 1)
    model = VAE(input_dim=1, latent_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    print(" Training VAE to learn 'Normal' traffic patterns...")
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        recon, mu, logvar = model(normal_data)
        loss = loss_function(recon, normal_data, mu, logvar)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")
    print(" Detecting anomalies in the full dataset...")
    model.eval()
    with torch.no_grad():
        reconstructed, mu, log_var = model(data_tensor)
        errors = torch.abs(data_tensor - reconstructed).numpy()
    threshold = np.percentile(errors, 98)
    df['anomaly_score'] = errors
    df['is_anomaly'] = df['anomaly_score'] > threshold
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['vehicle_count'], label='Actual Traffic', color='blue', alpha=0.6)
    anomalies = df[df['is_anomaly'] == True]
    plt.scatter(anomalies.index, anomalies['vehicle_count'], color='red', label='Detected Anomaly', zorder=5)
    plt.title("Traffic Anomaly Detection (VAE Model)")
    plt.xlabel("Time (Hours)")
    plt.ylabel("Vehicle Count")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    output_plot = "output_anomaly_test.png"
    plt.savefig(output_plot)
    print(f" Detection complete. Results saved to: {output_plot}")
    plt.show()
    print("\n--- Detected Notable Anomalies ---")
    print(anomalies[['timestamp', 'vehicle_count', 'anomaly_score']].head(10))

if __name__ == "__main__":
    run_anomaly_detection()
