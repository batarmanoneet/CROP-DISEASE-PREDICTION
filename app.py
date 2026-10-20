import os
from pathlib import Path

import streamlit as st
from PIL import Image

from src.chatbot import generate_chat_response, is_chat_api_configured
from src.knowledge_base import KnowledgeBase
from src.predictor import PlantDiseasePredictor

st.set_page_config(
    page_title="Crop Doctor AI",
    page_icon="🌱",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_ID = os.getenv("PLANT_DISEASE_MODEL", "JohnSupertramp2026/convnext-tiny-plantvillage")

@st.cache_resource(show_spinner="Loading the plant-disease model…")
def get_predictor():
    return PlantDiseasePredictor(MODEL_ID)

@st.cache_resource

def get_kb():
    return KnowledgeBase(str(BASE_DIR / "data" / "knowledge_base.json"))

st.title("🌱 Crop Doctor AI")
st.caption("Plant-disease image screening + evidence-grounded agriculture chat")

with st.sidebar:
    st.header("1. Crop photo")
    uploaded = st.file_uploader(
        "Upload a clear leaf/plant image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Close, well-lit photos work better. This system is a screening aid, not a definitive diagnosis.",
    )
    crop_hint = st.selectbox(
        "Crop (optional)",
        ["Auto-detect", "Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach", "Pepper", "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato"],
    )
    top_k = st.slider("Show top predictions", 2, 5, 3)

    st.divider()
    st.header("2. Ask")
    st.info("Try: ‘What should I check next?’ or ‘How can I reduce disease spread?’")
    if is_chat_api_configured():
        st.success("AI chat service: enabled")
    else:
        st.caption("Local chat mode: the image detector works without an API key. Add OPENAI_API_KEY in app Secrets to enable full AI chat; API usage may incur charges.")

kb = get_kb()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None

col1, col2 = st.columns([1, 1.35])

with col1:
    st.subheader("Photo")
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)
        st.session_state.image_bytes = uploaded.getvalue()

        if st.button("🔎 Analyze image", type="primary", use_container_width=True):
            try:
                with st.spinner("Loading the model and analyzing the image… The first run may take a few minutes."):
                    predictor = get_predictor()
                    results = predictor.predict(image, top_k=top_k)
                    analysis = results[0]
                    crop = analysis["crop"]
                    disease = analysis["disease"]
                    knowledge = kb.lookup(crop, disease)
                    st.session_state.analysis = {
                        "results": results,
                        "knowledge": knowledge,
                        "crop_hint": crop_hint,
                    }
            except Exception:
                st.error("The disease model could not be loaded or the image could not be analyzed. Check the app logs and your internet connection, then try again. The first model download can take a few minutes.")

    if st.session_state.analysis:
        a = st.session_state.analysis
        primary = a["results"][0]
        st.subheader("Prediction")
        confidence = primary["confidence"]
        if confidence >= 0.80:
            st.success(f"Likely: **{primary['pretty_disease']}**")
        elif confidence >= 0.50:
            st.warning(f"Possible: **{primary['pretty_disease']}**")
        else:
            st.error("Low-confidence result — retake the photo before acting on it.")

        st.metric("Model confidence", f"{confidence * 100:.1f}%")
        st.write(f"**Crop:** {primary['crop']}")

        st.write("**Top alternatives**")
        for r in a["results"]:
            st.write(f"• {r['pretty_disease']} — {r['confidence'] * 100:.1f}%")

        k = a["knowledge"]
        if k:
            st.write("**Knowledge-base notes**")
            st.write(k["summary"])
            if k.get("verify"):
                st.write("**Verify with:** " + k["verify"])
            if k.get("care"):
                st.write("**General care:** " + k["care"])
            if k.get("sources"):
                st.caption("Sources: " + " · ".join(k["sources"]))
        else:
            st.info("No detailed local entry exists for this class yet. The chat layer will give a cautious, general explanation and ask for more context.")
    else:
        st.markdown("**Tip:** Photograph one affected leaf close-up, with the symptoms filling most of the frame.")

with col2:
    st.subheader("Chat with Crop Doctor")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask about the plant, symptoms, prevention, or next checks…")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        context = ""
        if st.session_state.analysis:
            a = st.session_state.analysis
            p = a["results"][0]
            context = (
                f"Latest image screening: crop={p['crop']}; likely class={p['pretty_disease']}; "
                f"confidence={p['confidence']:.3f}; alternatives="
                + ", ".join(f"{x['pretty_disease']} ({x['confidence']:.2f})" for x in a["results"][1:])
            )
            if a["knowledge"]:
                context += f"\nKnowledge base: {a['knowledge']}"

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                response = generate_chat_response(
                    question=prompt,
                    conversation=st.session_state.messages,
                    context=context,
                    image_bytes=st.session_state.image_bytes,
                )
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

with st.expander("⚠️ Responsible-use note"):
    st.write(
        "The vision model is trained on PlantVillage-style imagery and should be treated as an image-screening tool. "
        "Field photographs can be substantially harder than clean dataset images. Confirm uncertain cases with a local agriculture professional or extension service before making treatment decisions."
    )
