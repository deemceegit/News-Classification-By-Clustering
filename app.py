import os
import re
import json
import unicodedata
import joblib
import spacy
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords

st.set_page_config(page_title="AI Research Topic Assistant", layout="centered")
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1557683316-973673baf926");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#NLP resources  

@st.cache_resource
def load_nlp_resources():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    stop_words = set(stopwords.words("english"))
    KEEP_SHORT = {"ai", "ml", "dl", "rl", "nlp"}

    try:
        nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        st.error("Missing Spacy Model. Run: python -m spacy download en_core_web_sm")
        st.stop()

    return stop_words, KEEP_SHORT, nlp


stop_words, KEEP_SHORT, nlp = load_nlp_resources()
artifacts_dir = "artifacts_clustering"

#w2v 

@st.cache_resource
def load_w2v_core():
    w2v_path = os.path.join(artifacts_dir, "word2vec.model")

    if os.path.exists(w2v_path):
        from gensim.models import Word2Vec
        return Word2Vec.load(w2v_path)

    return None


w2v_core_model = load_w2v_core()

#metadata load

@st.cache_data
def load_cluster_metadata():
    mapping_path = os.path.join(artifacts_dir, "cluster_mapping.json")

    mapping = {}

    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            mapping = json.load(f)

    return mapping


cluster_mapping = load_cluster_metadata()

#preprocess

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\n", " ").replace("\t", " ").strip()
    text = re.sub(r"(http\S+|www\.\S+)", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^A-Za-z0-9\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def preprocess_doc_lemma_only(doc) -> str:
    out = []

    for tok in doc:
        if tok.is_space or tok.text == "-":
            continue

        if tok.like_num:
            out.append("<num>")
            continue

        if not tok.is_alpha:
            continue

        lemma = tok.lemma_.lower().strip()

        if lemma == "datum":
            lemma = "data"

        if not lemma or tok.is_stop or lemma in stop_words:
            continue

        if len(lemma) < 2 and lemma not in KEEP_SHORT:
            continue

        out.append(lemma)

    return " ".join(out)

#load models

@st.cache_resource
def load_models():
    files = os.listdir(artifacts_dir)

    models = [
        f for f in files
        if f.endswith(".joblib") and ("BoW" in f or "TFIDF" in f or "W2V" in f)
    ]

    models.sort()

    return models


@st.cache_resource
def load_pipeline(model_name):
    return joblib.load(os.path.join(artifacts_dir, model_name))


available_models = load_models()

#choose-bar

st.divider()

col1, col2, col3 = st.columns([2,1,1])

with col1:
    selected_model = st.selectbox(
        "Model",
        available_models
    )

with col2:
    pipeline = load_pipeline(selected_model)
    st.write("Algorithm")
    st.code(pipeline["algorithm"])

with col3:
    st.write("Embedding")
    st.code(pipeline["embedding_name"])

st.divider()

#ui

st.title("AI Research Topic Assistant")
st.caption("Paste an arXiv abstract and the system will predict its research topic.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# show previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#user intput

user_input = st.chat_input("Paste an abstract...")

if user_input:

    # show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("Analyzing abstract..."):

            txt_cleaned = clean_text(user_input)
            final_text = preprocess_doc_lemma_only(nlp(txt_cleaned))

            emb_name = pipeline.get("embedding_name", "")
            ml_model = pipeline["model"]

            if "BoW" in emb_name or "TFIDF" in emb_name:

                vectorizer = pipeline["preprocess"]["vectorizer"]
                svd_50 = pipeline["preprocess"]["svd_50"]

                vec_sparse = vectorizer.transform([final_text])
                vec_final = svd_50.transform(vec_sparse)

            elif "W2V" in emb_name:

                tokens = final_text.split()

                vecs = [
                    w2v_core_model.wv[w]
                    for w in tokens
                    if w in w2v_core_model.wv
                ]

                if not vecs:
                    doc_vec = np.zeros(
                        w2v_core_model.vector_size,
                        dtype=np.float32
                    )
                else:
                    doc_vec = np.mean(vecs, axis=0).astype(np.float32)

                space_used = pipeline.get("meta", {}).get("space_used", "X_k")

                if space_used == "X_k":
                    vec_final = [doc_vec]
                else:
                    pca_50 = pipeline["preprocess"]["pca_50"]
                    vec_final = pca_50.transform([doc_vec])

            vec_norm = pipeline["normalizer"].transform(vec_final)

            cluster_id = ml_model.predict(vec_norm)[0]

            topic_name = cluster_mapping.get(
                str(cluster_id),
                f"Cluster {cluster_id}"
            )

            response = f"""
### Predicted Research Topic

**{topic_name}**

Cluster ID: `{cluster_id}`  
Embedding: `{emb_name}`  
"""

            st.markdown(response)

            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )