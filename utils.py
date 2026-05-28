import os
import faiss
import pickle
import warnings
import nltk
import numpy as np

import torch
import torch.nn as nn

from torchvision import transforms
from PIL import Image

from transformers import ViTForImageClassification
from sentence_transformers import SentenceTransformer

from groq import Groq
from api import API_KEY

warnings.filterwarnings("ignore")
nltk.download("punkt", quiet=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── CONFIG ────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.35
models = {}

# ── MODEL ARCHITECTURE ────────────────────────────────────────────────
def load_models():
    global models
    if models:
        return models
    
    vit_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=1,
        ignore_mismatched_sizes=True)
    vit_model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(vit_model.config.hidden_size, 1))
    vit_model.load_state_dict(
        torch.load(
            "vit_model.pth",
            map_location=DEVICE))
    vit_model.to(DEVICE)
    vit_model.eval()

    # ── FAISS RETRIEVAL INDEX ─────────────────────────────────────────────
    embedder = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    index = faiss.read_index("faiss_index.bin")
    with open("faiss_texts.pkl", "rb") as f:
        documents = pickle.load(f)
    with open("faiss_embeddings.pkl", "rb") as f:
        documents_embeddings = pickle.load(f)
        
    # ── GROQ CLIENT ───────────────────────────────────────────────────────
    GROQ_CLIENT = Groq(api_key=API_KEY)
    
    models = {
        "vit_model": vit_model,
        "embedder": embedder,
        "index": index,
        "documents": documents,
        "documents_embeddings": documents_embeddings,
        "groq_client": GROQ_CLIENT}

    return models

# ── IMAGE TRANSFORM ───────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225])])

# ── PREDICTION FUNCTIONS ──────────────────────────────────────────────
def predict_img(img):
    vit_model = models["vit_model"]
    
    img = img.convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = vit_model(img).logits
        prob = torch.sigmoid(outputs).item()
    return prob

def get_risk_level(prob):
    if prob < 0.30: return "Low"
    elif prob < 0.70: return "Moderate"
    else: return "High"
    
def retrieve_mmr(query, top_k=3, fetch_k=10, lambda_mult=0.7):
    embedder = models["embedder"]
    index = models["index"]
    documents = models["documents"]
    document_embeddings = models["documents_embeddings"]
    
    query_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    fetch_k = min(fetch_k, len(documents))
    distances, indices = index.search(query_emb, fetch_k)
    
    candidates, candidate_embs = [], []
    for score, idx in zip(distances[0], indices[0]):
        if score < SIMILARITY_THRESHOLD:
            continue
        if 0 <= idx < len(documents):
            candidates.append(documents[idx])
            candidate_embs.append(document_embeddings[idx])

    if len(candidates) == 0:
        return ["Irregular lesion morphology and evolving pigmentation patterns are clinically important indicators requiring dermatological assessment."]
    
    candidate_embs = np.array(candidate_embs)
    
    selected_docs = []
    selected_embs = []
    for _ in range(min(top_k, len(candidates))):
        if len(selected_docs) == 0:
            relevance_scores = candidate_embs @ query_emb[0]
            best_idx = int(np.argmax(relevance_scores))
        else:
            relevance_scores = candidate_embs @ query_emb[0]
            redundancy_scores = np.max(candidate_embs @ np.array(selected_embs).T, axis=1)
            mmr_scores = lambda_mult * relevance_scores - (1 - lambda_mult) * redundancy_scores
            best_idx = int(np.argmax(mmr_scores))
        
        selected_docs.append(candidates[best_idx])
        selected_embs.append(candidate_embs[best_idx])

        candidate_embs = np.delete(candidate_embs, best_idx, axis=0)
        candidates.pop(best_idx)
        if len(candidates) == 0:
            break
        
    return selected_docs
    
def build_prompt(prediction_prob, symptoms, retrieved_docs):
    risk = get_risk_level(prediction_prob)
    context = "\n".join([f"- {doc}" for doc in retrieved_docs])
    risk_instruction = {
        "High": (
            "This lesion MUST be described as clinically suspicious "
            "for melanoma/high malignancy risk. "
            "The explanation MUST support urgent dermatologist evaluation."),
        "Moderate": (
            "This lesion MUST be described as indeterminate/moderate risk. "
            "The explanation MUST support professional evaluation and monitoring."),
        "Low": (
            "This lesion MUST be described as likely benign/low risk. "
            "The explanation MUST avoid alarming language.")
    }[risk]
    
    prompt = f"""
Medical Knowledge Base: 
{context}

Patient Information:
- Symptoms: {symptoms.strip()}
- Malignancy Probability: {prediction_prob:.2f}
- Risk Level: {risk}

CRITICAL INSTRUCTION: {risk_instruction}

RULES: 
1. The response MUST remain consistent with the risk level.
2. Do NOT contradict the malignancy probability.
3. Do NOT invent diagnoses unsupported by the risk level.
4. Generate EXACTLY one sentence.
5. Do NOT use conversational filler.
6. Do NOT mention images or AI models.

Generate the clinical explanation.
"""
    return prompt

def generate_response(prompt, prediction_prob, max_retries=3):
    groq_client = models["groq_client"]
    risk = get_risk_level(prediction_prob)
    contradiction_terms = {
        "High": [
            "benign", "no concern", "normal",
            "not require", "no biopsy", "stable nevus"],
        "Low": ["melanoma", "highly malignant",
            "urgent biopsy", "aggressive cancer"]}
    system_message = (
        "You are a clinical decision-support assistant. "
        "You must strictly follow the provided malignancy risk level. "
        "Never contradict the probability score.")
    
    final_response = ""
    for attempt in range(max_retries):
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}],
            temperature=0.1 + (attempt * 0.1),
            max_tokens=80)
        response = completion.choices[0].message.content.strip()
        final_response = response
        bad_terms = contradiction_terms.get(risk, [])

        contradiction_found = any(term in response.lower() for term in bad_terms)
        if not contradiction_found:
            return response
    return final_response
    
def analyze(img, symptoms):
    if not models:
        load_models()
        
    prob = predict_img(img)
    retrieved_docs = retrieve_mmr(symptoms, top_k=3)
    prompt = build_prompt(prob, symptoms, retrieved_docs)
    explanation_output = generate_response(prompt, prob)
    if prob >= 0.70:
        condition = "Melanoma"
    elif prob < 0.30:
        condition = "Benign"
    else:
        condition = "Suspicious Lesion"
    
    full_report = (
        f"Predicted Condition: {condition}\n"
        f"Risk Level: {get_risk_level(prob)}\n"
        f"Explanation: {explanation_output}")
    
    return {
        "probability": prob,
        "risk_level": get_risk_level(prob),
        "retrieved_docs": retrieved_docs,
        "response": full_report}