import streamlit as st
import requests
import re

def format_assessment(text):
    condition = re.search(r"Predicted Condition:\s*(.*?)\s*Risk Level:", text)
    risk = re.search(r"Risk Level:\s*(.*?)\s*Explanation:", text)
    explanation = re.search(r"Explanation:\s*(.*)", text)
    
    condition_text = condition.group(1) if condition else "N/A"
    risk_text = risk.group(1) if risk else "N/A"
    explanation_text = explanation.group(1) if explanation else "N/A"
    
    if (condition_text == "N/A"
        or risk_text == "N/A"
        or explanation_text == "N/A"):
        return text
    
    return f"""
- **Predicted Condition:** {condition_text}
- **Risk Level:** {risk_text}
- **Explanation:** {explanation_text}"""
    
st.set_page_config(
    page_title="Skin Lesion Assistant",
    layout="centered"
)
st.title("Multimodal Skin Lesion Medical Assistant")
st.write("Upload a lesion image and describe symptoms.")

uploaded_file = st.file_uploader(
    "Upload Skin Lesion Image",
    type=["jpg", "jpeg", "png"]
)

symptoms = st.text_area("Describe symptoms")

if st.button("Analyze"):
    if uploaded_file is None:
        st.warning("Please upload an image.")
    elif not symptoms.strip():
        st.warning("Please enter symptoms.")
    else:
        with st.spinner("Running deep evaluation..."):
            files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            data = {"symptoms": symptoms}
            
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/analyze",
                    files=files,
                    data=data
                )
                response.raise_for_status()
                result = response.json()
                
                st.image(uploaded_file, caption="Uploaded Image", width='content')
                
                st.subheader("Prediction")
                prob_val = float(result["probability"])
                risk = result["risk_level"]
                assessment = result["response"]
                
                st.subheader("Result")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Melanoma Probability**: {prob_val:.4f}")
                    st.progress(min(max(prob_val, 0.0), 1.0))
                with col2:
                    st.write(f"**Risk Level**: {risk.upper()}")
                    
                st.subheader("Medical Assessment")
                st.markdown(format_assessment(assessment))
                
                st.subheader("Retrieved Evidence")
                for i, doc in enumerate(result["retrieved_docs"]):
                    with st.expander(f"Medical Document{i+1}", expanded=True):
                        st.write(doc)
                        
            except requests.exceptions.RequestException as e:
                st.error(f"Error communicating with FastAPI server: {e}")