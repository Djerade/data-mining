# Interface Streamlit pour explorer les segments de clients
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === Chargement du modèle et des objets sauvegardés ===
MODELE_PATH = "./kmeans_model.pkl"
SCALER_PATH = "./scaler.pkl"

if os.path.exists(MODELE_PATH) and os.path.exists(SCALER_PATH):
    kmeans = joblib.load(MODELE_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    st.error("❌ Les fichiers du modèle ou du scaler sont introuvables. Veuillez exécuter le script d'entraînement au préalable.")
    st.stop()

# === Chargement ou simulation des données ===
np.random.seed(42)
n_clients = 1000
data = pd.DataFrame({
    'client_id': range(1, n_clients + 1),
    'recence': np.random.randint(1, 365, n_clients),
    'frequence': np.random.randint(1, 50, n_clients),
    'panier_moyen': np.random.uniform(10, 1000, n_clients)
})

features = ['recence', 'frequence', 'panier_moyen']
X_scaled = scaler.transform(data[features])
data['cluster'] = kmeans.predict(X_scaled)

# === Interface Streamlit ===
st.set_page_config(page_title="Segmentation Clients", layout="wide")
st.title("📊 Segmentation des Clients - Data Mining")

st.markdown("Ce tableau de bord permet d'explorer les groupes de clients identifiés par clustering (K-Means) selon leurs comportements d'achat.")

# Filtrage par cluster
st.info(f"🔢 Nombre de segments (clusters) détectés : {len(data['cluster'].unique())}")
clusters = sorted(data['cluster'].unique())
selected_cluster = st.selectbox("Choisissez un segment de clients :", options=["Tous"] + list(clusters))

if selected_cluster != "Tous":
    filtered_data = data[data['cluster'] == selected_cluster]
else:
    filtered_data = data

# Statistiques descriptives
st.subheader("📌 Statistiques descriptives")
st.dataframe(filtered_data[features + ['cluster']].describe().round(2))

# Visualisation
st.subheader("📉 Visualisation des variables RFM")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=filtered_data, x='recence', y='panier_moyen', hue='cluster', palette='tab10', ax=ax)
plt.title("Recence vs Panier Moyen")
st.pyplot(fig)

# Autres graphiques
st.subheader("📈 Distribution des variables par cluster")

# Histogrammes
for col in features:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=filtered_data, x=col, hue='cluster', multiple='stack', palette='tab10', bins=30, ax=ax)
    ax.set_title(f"Distribution de {col}")
    st.pyplot(fig)

# Pairplot simplifié
st.subheader("📊 Visualisation croisée des variables")
fig = sns.pairplot(data=filtered_data, vars=features, hue='cluster', palette='tab10')
st.pyplot(fig)

# Téléchargement des données segmentées
csv = data.to_csv(index=False).encode('utf-8')
st.download_button("📥 Télécharger les données complètes", data=csv, file_name="clients_segmentes.csv", mime='text/csv')

# Cluster summary
st.subheader("📊 Moyenne par segment")
summary = data.groupby('cluster')[features].mean().round(2)
st.dataframe(summary)

# Description personnalisée des segments
st.subheader("🧠 Interprétation des segments")
descriptions = {
    0: "🟢 Cluster 0 : Clients fidèles avec panier élevé (🎯 Cible VIP)",
    1: "🔵 Cluster 1 : Acheteurs ponctuels mais récents",
    2: "🟡 Cluster 2 : Anciens clients inactifs",
    3: "🟠 Cluster 3 : Clients à petits paniers réguliers (bons candidats pour promotions)"
}

for cluster_id, desc in descriptions.items():
    st.markdown(f"**{desc}**")

# Interprétation des segments
st.subheader("🧠 Interprétation des segments")
cluster_labels = {
    0: "👑 Cluster 0 : Clients fidèles avec panier élevé (cible VIP)",
    1: "🕒 Cluster 1 : Acheteurs ponctuels mais récents",
    2: "❌ Cluster 2 : Anciens clients inactifs",
    3: "💸 Cluster 3 : Clients à petits paniers réguliers (bons candidats pour promotions)"
}

for cluster_id, label in cluster_labels.items():
    st.markdown(f"**{label}**")
