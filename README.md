# Crop Doctor AI 🌱

A Streamlit MVP for plant-disease image screening and a farmer-friendly chat assistant.

## Features

- Plant-disease classifier across 38 PlantVillage-style classes.
- Top predictions, model confidence, and basic disease notes.
- Local fallback chat that works without an API key.
- Optional OpenAI Responses API chat with image-aware follow-up.
- Streamlit Cloud deployment configuration and step-by-step guide.

## Run locally on Windows

1. Extract the ZIP.
2. Open Command Prompt inside the `crop_doctor` folder.
3. Run:

   ```bat
   python -m venv .venv
   call .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   streamlit run app.py
   ```

4. Open the URL printed by Streamlit, usually `http://localhost:8501`.

The first image analysis downloads the model from Hugging Face and may take a few minutes.

## Optional chat API setup

The model is optional: without `OPENAI_API_KEY`, the app still supports image screening and uses a built-in fallback chat. For full AI chat, configure `OPENAI_API_KEY` and optionally `OPENAI_MODEL` as environment variables or Streamlit secrets. API use may incur charges. Never commit actual credentials to GitHub.

## Deploy online

Follow [`DEPLOY_TO_CLOUD.md`](DEPLOY_TO_CLOUD.md) to publish the app on Streamlit Community Cloud so it can be opened from a phone or another computer without your laptop running.

## Model and limitations

Default model: `JohnSupertramp2026/convnext-tiny-plantvillage` (38-class ConvNeXt-Tiny). Its model card reports 96.41% accuracy and 0.919 macro-F1 on its stated PlantVillage split, but this does not guarantee performance on real-world field photographs. This is a screening aid, not a definitive diagnosis.

## Test

```bash
python -m compileall -q app.py src tests
pytest -q  # optional; install pytest first if needed
```
