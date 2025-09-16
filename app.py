import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# ---- Configs ----
CSV_PATH_DEFAULT = "data/imdb_top250_clusters.csv"
VECTORIZER_PATH_DEFAULT = "models/tfidf_vectorizer.pkl"
TFIDF_PARAMS = dict(sublinear_tf=True, min_df=0.05, max_df=0.95, ngram_range=(1, 2))

# ---- Estilo Customizado ----
CUSTOM_CSS = """
<style>
    .title { font-size: 32px; font-weight: 700; color: #ff4b4b; }
    .subheader { font-size: 20px; font-weight: 600; margin-top: 20px; }
    .card {
        background-color: #f8f9fa;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.08);
        margin-bottom: 15px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---- Utilitários ----
@st.cache_data(show_spinner=False)
def ensure_nltk_stopwords():
    try:
        _ = stopwords.words('portuguese')
    except Exception:
        nltk.download('stopwords')
    return set(stopwords.words('portuguese'))

PT_STOPWORDS = ensure_nltk_stopwords()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[\W_]+", " ", text)
    tokens = [t for t in text.split() if t and t not in PT_STOPWORDS]
    return " ".join(tokens)

@st.cache_data(show_spinner=False)
def load_dataframe(path: str = CSV_PATH_DEFAULT) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def get_vectorizer_and_matrix(df: pd.DataFrame, load_saved_vectorizer: bool = True,
                              vec_path: str = VECTORIZER_PATH_DEFAULT):
    vect = None
    if load_saved_vectorizer:
        try:
            vect = joblib.load(vec_path)
        except Exception:
            vect = None
    if vect is None:
        vect = TfidfVectorizer(**TFIDF_PARAMS)
        vect.fit(df['sinopse_no_stopwords'])
    X = vect.transform(df['sinopse_no_stopwords'])
    return vect, X

def compute_cluster_centroids(X, cluster_labels):
    """Calcula a média vetorial de cada cluster"""
    unique_clusters = np.unique(cluster_labels)
    centroids = []
    for c in unique_clusters:
        idx = np.where(cluster_labels == c)[0]
        centroid = X[idx].mean(axis=0)
        centroid = np.asarray(centroid).ravel()
        centroids.append(centroid)
    return unique_clusters, np.vstack(centroids)

# ---- Recomendações ----
def recommend_by_index(df, X, idx, top_k=5):
    cluster = df.loc[idx, 'cluster']
    members = df[df['cluster'] == cluster].index.to_numpy()
    sims = cosine_similarity(X[idx], X[members]).ravel()
    results = pd.DataFrame({'idx': members, 'sim': sims})
    results = results[results['idx'] != idx]
    results = results.sort_values('sim', ascending=False).head(top_k)
    recs = df.loc[results['idx']].copy()
    recs['similarity'] = results['sim'].values
    return recs.reset_index(drop=True)

def recommend_in_cluster(df, X, cluster_labels, cluster_centroids, user_vector, top_k=5):
    """Atribui a sinopse ao cluster mais próximo e retorna recomendações ordenadas por similaridade"""
    sims_to_centroids = cosine_similarity(user_vector, cluster_centroids).ravel()
    best_idx = np.argmax(sims_to_centroids)
    assigned_cluster = cluster_labels[best_idx]
    cluster_members = df[df['cluster'] == assigned_cluster].index.to_numpy()
    sims_to_members = cosine_similarity(user_vector, X[cluster_members]).ravel()
    
    results = pd.DataFrame({'idx': cluster_members, 'sim': sims_to_members})
    results = results.sort_values('sim', ascending=False).head(top_k)
    recs = df.loc[results['idx']].copy()
    recs['similarity'] = results['sim'].values
    return recs.reset_index(drop=True), assigned_cluster, sims_to_centroids[best_idx]

# ---- App UI ----
def main():
    st.set_page_config(page_title="🎬 Recomendador IMDB", layout="wide")
    st.markdown('<div class="title">🎬 Recomendador de Filmes (IMDB Top)</div>', unsafe_allow_html=True)

    df = load_dataframe(CSV_PATH_DEFAULT)
    vect, X = get_vectorizer_and_matrix(df, load_saved_vectorizer=True)
    clusters_unique, centroids = compute_cluster_centroids(X, df['cluster'].to_numpy())

    method = st.radio("📌 Escolha o método:", ["🔹 Método 1 — Escolher sinopse", "🔹 Método 2 — Escrever sinopse"])
    st.divider()

    # ----- MÉTODO 1 -----
    if method.startswith("🔹 Método 1"):
        st.markdown('<div class="subheader">📖 Método 1 — escolha entre 3-5 sinopses</div>', unsafe_allow_html=True)
        n_options = st.slider("Quantas opções mostrar?", 3, 5, 4)
        seed = st.number_input("Semente (aleatoriedade)", value=42, step=1)
        sample_df = df.sample(n=n_options, random_state=int(seed))

        mapping = {}
        for i, (idx, row) in enumerate(sample_df.iterrows()):
            label = f"Opção {i+1}"
            preview = row['sinopse'][:400] + ("..." if len(row['sinopse']) > 400 else "")
            st.markdown(f'<div class="card"><b>{label}</b><br>{preview}</div>', unsafe_allow_html=True)
            mapping[label] = idx

        choice = st.radio("Qual sinopse mais te agrada?", list(mapping.keys()))
        if st.button("✨ Gerar recomendações"):
            idx_chosen = mapping[choice]
            recs = recommend_by_index(df, X, idx_chosen, top_k=5)
            if recs.empty:
                st.info("Nenhum filme encontrado neste cluster.")
            else:
                st.markdown("### 🎥 Recomendações")
                for _, r in recs.iterrows():
                    st.markdown(
                        f"""<div class="card">
                        <b>{r['title']}</b> ({r.get('year','?')}) — {r.get('genre','')}
                        <br><i>Similaridade:</i> {r['similarity']:.3f}
                        <br><br><i>Sinopse:</i> {r['sinopse']}
                        </div>""",
                        unsafe_allow_html=True
                    )

    # ----- MÉTODO 2 -----
    else:
        st.markdown('<div class="subheader">✍️ Método 2 — escreva uma sinopse</div>', unsafe_allow_html=True)
        user_text = st.text_area("Digite uma sinopse curta (2–6 frases)")
        top_k = st.slider("Quantas recomendações mostrar?", 1, 10, 5)
        if st.button("✨ Submeter e recomendar"):
            if not user_text.strip():
                st.warning("Escreva algo antes de submeter.")
            else:
                user_clean = clean_text(user_text)
                vec = vect.transform([user_clean])
                recs, assigned_cluster, cluster_sim = recommend_in_cluster(
                    df, X, clusters_unique, centroids, vec, top_k
                )
                st.success(f"📌 Cluster atribuído: {assigned_cluster} — Similaridade ao centróide: {cluster_sim:.3f}")
                if recs.empty:
                    st.info("Não há recomendações para este cluster.")
                else:
                    st.markdown("### 🎥 Recomendações")
                    for _, r in recs.iterrows():
                        st.markdown(
                            f"""<div class="card">
                            <b>{r['title']}</b> ({r.get('year','?')}) — {r.get('genre','')}
                            <br><i>Similaridade:</i> {r['similarity']:.3f}
                            <br><br><i>Sinopse:</i> {r['sinopse'][:500]}...
                            </div>""",
                            unsafe_allow_html=True
                        )

if __name__ == "__main__":
    main()
