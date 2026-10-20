import os
from pathlib import Path

import streamlit as st
from PIL import Image

from src.chatbot import generate_chat_response, is_chat_api_configured
from src.knowledge_base import KnowledgeBase
from src.predictor import PlantDiseasePredictor

st.set_page_config(page_title="Crop Doctor AI", page_icon="🌱", layout="wide")

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
        help="A close, well-lit photo of one affected leaf is best. The result is a screening aid, not a diagnosis.",
    )
    crop_hint = st.selectbox(
        "Crop (if known)",
        ["Auto-detect", "Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach", "Pepper", "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato"],
    )
    top_k = st.slider("Show top predictions", 2, 5, 3)
    st.caption("When you select a crop, the app shows candidates from that crop too. Their scores are not increased or renormalised.")
    st.divider()
    st.header("2. Ask")
    st.info("Try: ‘What should I check next?’ or ‘How can I reduce disease spread?’")
    if is_chat_api_configured():
        st.success("AI chat service: enabled")
    else:
        st.caption("Local chat mode is active. Add OPENAI_API_KEY in app Secrets to enable full AI chat; API usage may incur charges.")

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
                    prediction = predictor.predict(
                        image,
                        top_k=top_k,
                        crop_hint=crop_hint,
                    )
                    global_top = prediction["global_top"]
                    displayed_results = prediction["results"]

                    # Don't show disease-specific notes if the model is uncertain or
                    # the overall prediction conflicts with the user's selected crop.
                    score_is_low = global_top["confidence"] < 0.30
                    crop_conflict = bool(
                        prediction["crop_hint"]
                        and global_top["crop"].casefold().startswith(prediction["crop_hint"].casefold()) is False
                        and not prediction["crop_hint"].casefold().startswith(global_top["crop"].casefold())
                    )
                    knowledge = None
                    if not score_is_low and not crop_conflict:
                        p = displayed_results[0]
                        knowledge = kb.lookup(p["crop"], p["disease"])

                    st.session_state.analysis = {
                        "results": displayed_results,
                        "global_results": prediction["global_results"],
                        "global_top": global_top,
                        "knowledge": knowledge,
                        "crop_hint": prediction["crop_hint"],
                        "crop_hint_applied": prediction["crop_hint_applied"],
                        "crop_hint_has_classes": prediction["crop_hint_has_classes"],
                        "crop_conflict": crop_conflict,
                        "score_is_low": score_is_low,
                    }
            except Exception as exc:
                st.error("The disease model could not be loaded or the image could not be analyzed. Check the app logs and your internet connection, then try again. The first model download can take a few minutes.")
                with st.expander("Technical details"):
                    st.code(f"{type(exc).__name__}: {exc}")

    if st.session_state.analysis:
        a = st.session_state.analysis
        global_top = a["global_top"]
        st.subheader("Prediction review")
        st.metric("Overall top class score", f"{global_top['confidence'] * 100:.1f}%")
        st.write(f"**Model's overall top class:** {global_top['pretty_disease']}")
        st.caption("This is the model's raw softmax score, not a calibrated probability of disease.")

        if a["score_is_low"]:
            st.error("Inconclusive result — the model's top score is low. Do not act on disease or treatment suggestions from this image.")
        if a["crop_conflict"]:
            st.warning(
                f"You selected {a['crop_hint']}, but the model's overall top class is {global_top['crop']}. "
                "This conflict means the result is uncertain; the crop-specific list below is only a set of candidates, not a diagnosis."
            )

        if a["crop_hint"] and a["crop_hint_applied"]:
            st.write(f"**Candidates for the selected crop: {a['crop_hint']}**")
            st.caption("Scores remain from the full model output. They have not been renormalised to make them look higher.")
        elif a["crop_hint"] and not a["crop_hint_has_classes"]:
            st.warning("The model label list did not contain classes matching the selected crop. Showing overall predictions instead.")
        else:
            st.write("**Overall top alternatives (all supported crops)**")

        for r in a["results"]:
            st.write(f"• {r['pretty_disease']} — raw model score {r['confidence'] * 100:.1f}%")

        if a["knowledge"]:
            k = a["knowledge"]
            st.write("**Knowledge-base notes**")
            st.write(k["summary"])
            if k.get("verify"):
                st.write("**Verify with:** " + k["verify"])
            if k.get("care"):
                st.write("**General care:** " + k["care"])
            if k.get("sources"):
                st.caption("Sources: " + " · ".join(k["sources"]))
        else:
            st.info("No disease-specific advice is shown because the result is uncertain or does not match the selected crop. Retake the photo or ask a local agriculture professional before deciding on treatment.")
    else:
        st.markdown("**Tip:** Photograph one affected leaf close-up, with symptoms filling most of the frame. If you know the crop, select it before clicking Analyze image.")

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
            p = a["global_top"]
            context = (
                f"IMPORTANT: This is an uncertain image-screening result, not a confirmed diagnosis. "
                f"Global top class={p['pretty_disease']}; raw softmax score={p['confidence']:.3f}; "
                f"user-selected crop={a['crop_hint'] or 'not specified'}; crop conflict={a['crop_conflict']}. "
                "Do not present a disease as certain or give chemical/pesticide dosage advice. Ask for more observations or recommend local agricultural extension assistance when appropriate.\n"
                "Displayed candidates: "
                + ", ".join(f"{x['pretty_disease']} (raw score {x['confidence']:.3f})" for x in a["results"])
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
        "The vision model was trained on PlantVillage-style imagery. It may fail on ordinary field/phone photos, unfamiliar crops, mixed symptoms, nutrient problems, and pests. "
        "Softmax scores are not calibrated disease probabilities. Treat uncertain or conflicting results as inconclusive and confirm consequential decisions with a local agricultural professional or extension service."
    )
