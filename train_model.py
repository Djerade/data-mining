
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# === Génération de données simulées pour l'entraînement ===
np.random.seed(42)
n_clients = 1000
data = pd.DataFrame({
    'recence': np.random.randint(1, 365, n_clients),
    'frequence': np.random.randint(1, 50, n_clients),
    'panier_moyen': np.random.uniform(10, 1000, n_clients)
})

features = ['recence', 'frequence', 'panier_moyen']

# === Normalisation ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

# === Entraînement du modèle KMeans avec 4 clusters ===
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
kmeans.fit(X_scaled)

# === Sauvegarde du modèle et du scaler ===
joblib.dump(kmeans, "kmeans_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Modèle KMeans et scaler sauvegardés avec succès.")
