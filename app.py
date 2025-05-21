import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import os

# === Load your datasets ===
@st.cache_data
def load_dataset():
    file_id = st.secrets["file_id"]
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "combined_bgg_dataset.csv"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    return pd.read_csv(output)

df = load_dataset()

def load_dataset_fi():
    file_fi_id = st.secrets["file_fi_id"]
    url = f"https://drive.google.com/uc?id={file_fi_id}"
    output = "feature_importance_named.csv"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    return pd.read_csv(output)

feature_importance_df = load_dataset_fi()

# === Prepare columns ===
mechanic_cols = [col for col in df.columns if col.startswith('mechanic_')]
designer_cols = [col for col in df.columns if col not in mechanic_cols + ['BGGId', 'Games_Name', 'Low-Exp Designer']]
feature_cols = mechanic_cols + designer_cols

# === Create feature weights ===
feature_weights = {
    f"mechanic_{int(row['ID'])}": row['importance']
    for _, row in feature_importance_df.iterrows()
}
for col in feature_cols:
    if col not in feature_weights:
        feature_weights[col] = 1.0

st.info(f"🎲 Loaded {len(df)} games, {len(mechanic_cols)} mechanics, and {len(designer_cols)} designers")

# === User Inputs ===
use_feature_importance = st.checkbox("Use mechanic feature importance?", value=True)
designer_weight = st.slider("Designer weight multiplier", 0.0, 0.1, 0.01)
game_names_input = st.text_input("Enter game names (comma-separated)", value="Terraforming Mars")

# === Recommendation Function ===
def recommend_games(input_games, df, feature_cols, weights_array, top_n=20, batch_size=500):
    input_vectors = df[df['Games_Name'].isin(input_games)][feature_cols].fillna(0).astype(np.float32).values
    if len(input_vectors) == 0:
        st.error(f"❌ Could not find any of the games: {input_games}")
        return pd.DataFrame()

    user_vec = np.mean(input_vectors, axis=0)
    user_vec_w = np.nan_to_num(user_vec * weights_array.astype(np.float32))

    similarities = np.zeros(len(df), dtype=np.float32)

    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch = df.iloc[start:end][feature_cols].fillna(0).astype(np.float32).values
        batch = np.nan_to_num(batch * weights_array)
        sims = cosine_similarity([user_vec_w], batch).flatten()
        similarities[start:end] = sims

    result_df = df[['Games_Name']].copy()
    result_df['similarity'] = similarities
    result_df = result_df[~df['Games_Name'].isin(input_games)]
    result_df = result_df.sort_values('similarity', ascending=False).head(top_n)

    return result_df

# === Recommendation Trigger ===
if st.button("Get Recommendations"):
    input_games = [g.strip() for g in game_names_input.split(",") if g.strip()]

    adjusted_weights = feature_weights.copy()
    for designer in designer_cols:
        adjusted_weights[designer] *= designer_weight

    weights_array = np.array([
        adjusted_weights.get(col, 1.0) if (use_feature_importance or col in designer_cols) else 1.0
        for col in feature_cols
    ], dtype=np.float32)

    df_filled = df.fillna(0)

    recommendations = recommend_games(input_games, df_filled, feature_cols, weights_array)

    if not recommendations.empty:
        st.subheader("✅ Top Recommendations:")
        st.dataframe(recommendations.style.format({'similarity': '{:.3f}'}))
    else:
        st.warning("No recommendations found.")
