import faiss
import pickle
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification
from sentence_transformers import SentenceTransformer
from groq import Groq
from api import API_KEY

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── MODEL ARCHITECTURE ────────────────────────────────────────────────
models = {}

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
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index = faiss.read_index("faiss_index.bin")
    with open("faiss_texts.pkl", "rb") as f:
        documents = pickle.load(f)
        
    # ── GROQ CLIENT ───────────────────────────────────────────────────────
    GROQ_CLIENT = Groq(api_key=API_KEY)
    
    models = {
        "vit_model": vit_model,
        "embedder": embedder,
        "index": index,
        "documents": documents,
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

def retrieve(query, top_k=3):
    embedder = models["embedder"]
    index = models["index"]
    documents = models["documents"]
    
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        if 0 <= idx < len(documents):
            results.append(documents[idx])
    return results

def get_risk_level(prob):
    if prob < 0.30: return "Low"
    elif prob < 0.70: return "Moderate"
    else: return "High"
    
def build_prompt(prediction_prob, symptoms, retrieved_docs):
    context = "\n".join([f"- {doc}" for doc in retrieved_docs])
    risk = get_risk_level(prediction_prob)
    
    prompt = (
        f"Medical Knowledge Base Reference:\n{context}\n\n"
        f"Patient Diagnostic Metrics:\n"
        f"- Reported Symptoms: {symptoms.strip()}\n"
        f"- Vision Classification Probability Score: {prediction_prob:.4f}\n"
        f"- Assigned Risk Category: {risk}\n\n"
        f"Please construct the explanation portion of the report.")
    return prompt

def generate_response(prompt):
    groq_client = models["groq_client"]
    system_message = (
        "You are an expert clinical decision-support assistant. Your task is to generate a concise, "
        "one-sentence clinical explanation for a patient report. Use the provided Medical Knowledge Base Reference. "
        "Do not hallucinate, do not refer to figures/images, and do not append conversational filler text.")
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        max_tokens=80)
    return completion.choices[0].message.content.strip()
    
def analyze(img, symptoms):
    if not models:
        load_models()
        
    prob = predict_img(img)
    retrieved_docs = retrieve(symptoms, top_k=3)
    prompt = build_prompt(prob, symptoms, retrieved_docs)
    explanation_output = generate_response(prompt)
    
    full_report = (
        f"Predicted Condition: {'Melanoma' if prob >= 0.5 else 'Benign'}\n"
        f"Risk Level: {get_risk_level(prob)}\n"
        f"Explanation: {explanation_output}")
    
    return {
        "probability": prob,
        "risk_level": get_risk_level(prob),
        "retrieved_docs": retrieved_docs,
        "response": full_report}