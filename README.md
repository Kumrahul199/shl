import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_shl_catalog(path='data/shl_catalog.json'):
    with open(path) as f:
        return json.load(f)

def build_index(data, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [item["description"] for item in data]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))
    return index, embeddings, model
