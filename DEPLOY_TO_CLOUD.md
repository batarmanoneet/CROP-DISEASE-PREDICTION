# Put Crop Doctor AI online with Streamlit Community Cloud

This guide publishes the app at a shareable `https://....streamlit.app` address. Your laptop does not need to remain on after deployment.

## Before starting

- A GitHub account.
- A Streamlit Community Cloud account, signed in with GitHub.
- Internet access for the first model download.
- An OpenAI API key is **optional**. Without it, disease-image screening still works and the chat panel uses a small built-in fallback. Full AI chat uses the OpenAI API and may cost money depending on your account usage.

## Step 1 — Extract the project

1. Download the ZIP from ChatGPT.
2. Right-click it in Windows File Explorer and select **Extract All...**.
3. Open the extracted `crop_doctor` folder. You should see `app.py`, `requirements.txt`, `data`, and `src` directly inside it.

Do not upload the ZIP itself to GitHub. Upload the contents of the `crop_doctor` folder.

## Step 2 — Create a GitHub repository

1. Open [GitHub](https://github.com/) and create a new repository named `crop-doctor-ai`.
2. For the simplest first deployment, make the repository **Public**. A public repository means the source code is visible to others; never add API keys, passwords, or private data to it.
3. Choose the option to add/upload files.
4. In File Explorer, open the extracted `crop_doctor` folder, select all its files and folders (including `.streamlit`), and upload them to the repository. GitHub's web uploader may hide dot-folders on some systems; make sure `.streamlit/config.toml` is included if possible. The app can still run without this optional config file.
5. Commit the files to the `main` branch.

Make sure the repository's top level contains `app.py`, `requirements.txt`, `src/`, and `data/`. Do not upload `.venv/`, `.env`, or `.streamlit/secrets.toml`.

## Step 3 — Deploy the repository

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in and connect/authorize your GitHub account when asked.
3. Choose **Create app** / **Yup, I have an app**.
4. Select repository `crop-doctor-ai`, branch `main`, and main file path `app.py`.
5. In **Advanced settings**, choose **Python 3.12**.
6. Leave Secrets blank if you want the built-in fallback chat, or follow the optional section below to enable OpenAI chat.
7. Click **Deploy** and wait for the build to finish.

The first deploy installs the Python packages. The first image analysis downloads the vision model and can take a few minutes. Free hosting resources are limited, so if the build fails, open **Manage app / Logs** and review the first error message.

## Optional — Enable full AI chat safely

You need an OpenAI API key for this option. API use is separate from a ChatGPT subscription and may incur charges.

1. In Streamlit Community Cloud, open your app's menu/settings and find **Secrets**.
2. Add these lines, replacing the placeholder with your own key:

   ```toml
   OPENAI_API_KEY = "your_real_api_key_here"
   OPENAI_MODEL = "gpt-6-luna"
   ```

3. Save and let the app restart.
4. Never commit a real key into GitHub or share it in screenshots/chat messages. If a key is exposed, revoke it and create a new one.

## Step 4 — Test it

Open the Streamlit URL given after deployment on your laptop or phone. Test with a clear photo of a single affected leaf. Confirm that:

- the photo uploads;
- the model returns top predictions;
- the chatbot panel responds;
- low-confidence outputs are not treated as definitive diagnoses.

## Updates later

Edit or upload changed files to the same GitHub repository and commit them. Community Cloud normally redeploys from repository updates automatically.

## Common problems

- **Dependency/build error:** open the app logs. Cloud deployment uses a clean Linux environment, not your laptop's `.venv`.
- **Model load error:** wait for the initial model download and retry. If it repeats, use Logs to check memory/network/model issues.
- **Chat falls back to built-in replies:** this is expected without `OPENAI_API_KEY`, or if the configured API call fails. The image detector can still work.
- **URL works on your laptop only?** Make sure you opened the public `streamlit.app` URL, not `localhost`.

## Important limitations

The image model has been evaluated on PlantVillage-style data; ordinary field images may be harder. Predictions are not a verified agricultural diagnosis. Use regional agricultural-extension guidance for consequential treatment decisions, and do not rely on this prototype alone for pesticide selection or dosage.
