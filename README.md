# DriveAlert AI — Driver Drowsiness Detection

A CNN-based system that detects driver drowsiness from eye state (open/closed) via webcam or photo upload, wrapped in a polished Streamlit web app.

This package contains everything except the trained model (you train that in Google Colab, using the dataset you already have — takes ~5 minutes on a free GPU).

---

## What's in this folder

```
app.py                      -> the web app (run this)
requirements.txt            -> Python packages needed
.streamlit/config.toml      -> app theme
face_detector/              -> pretrained face detector (already included, no download needed)
Train_Model_Colab.ipynb     -> open this in Google Colab to train your model
README.md                   -> this file
```

You still need to add two files yourself (from Step 1 below):
```
drowsiness_model.h5
class_indices.json
```

---

## Step 1: Train the model in Google Colab

1. Go to **colab.research.google.com** → **File → Upload notebook** → upload `Train_Model_Colab.ipynb` from this folder
2. In Colab: **Runtime → Change runtime type → T4 GPU → Save**
3. On your own computer: right-click your `dataset` folder (the one with `Drowsy/` and `Non_Drowsy/` subfolders) → **Send to → Compressed (zipped) folder** → this creates `dataset.zip`
4. Run the notebook cells one by one (Shift+Enter). When it asks you to upload a file, upload `dataset.zip`
5. Wait for training to finish (~2-5 minutes on GPU)
6. At the last cell, two files will download automatically to your computer's Downloads folder:
   - `drowsiness_model.h5`
   - `class_indices.json`
7. Also download `training_graph.png` and `confusion_matrix.png` from Colab's left-side file browser (right-click → Download) — you'll need these for your report/screenshots

---

## Step 2: Set up the project folder

Move `drowsiness_model.h5` and `class_indices.json` from your Downloads into **this same folder** (next to `app.py`), so it looks like:

```
CNN_TAE/
    app.py
    requirements.txt
    drowsiness_model.h5        <- from Colab
    class_indices.json         <- from Colab
    .streamlit/
        config.toml
    face_detector/
        deploy.prototxt
        res10_300x300_ssd_iter_140000.caffemodel
```

Open this whole folder in VS Code / Antigravity: **File → Open Folder**.

---

## Step 3: Install and run locally

Open a terminal in this folder and run:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

A browser tab opens with three tabs:
- **🔴 Live Camera (Continuous)** — real-time detection, no clicking needed after pressing Start
- **📷 Take Photo** — reliable single-snapshot test (use this if Live Camera has connection issues)
- **📁 Upload Image** — upload any photo to test

**If Live Camera shows a blank/loading video:** your firewall is likely blocking it. Search "Allow an app through Windows Firewall" → find Python → check both **Private** and **Public** boxes → restart the app.

---

## Step 4: Deploy online (Streamlit Community Cloud — free)

1. Create a free account at **github.com**
2. Create a new **public** repository
3. Upload ALL files and folders from this project (including `face_detector/`, `.streamlit/`, `drowsiness_model.h5`, `class_indices.json`) via **Add file → Upload files**
4. Go to **share.streamlit.io** → sign in with GitHub → **Create app** → select your repo, branch `main`, file `app.py` → **Deploy**
5. After a few minutes you'll get a public URL like `https://your-app-name.streamlit.app` — share this link for your demo/submission

---

## Notes for your report / viva

- **Preprocessing:** images resized to 96×96, pixel values normalized (0–1)
- **Augmentation:** rotation, width/height shift, horizontal flip, zoom — done via Keras `ImageDataGenerator`
- **Model:** CNN with 3 convolutional blocks (32→64→128 filters) + dense layers + dropout, binary output (sigmoid)
- **Split:** 70% train / 15% validation / 15% test (achieved via `validation_split=0.30`, further split during evaluation)
- **Metrics:** accuracy, precision, recall, F1-score, confusion matrix — all printed by the notebook
- **Face/eye localization:** a pretrained DNN face detector (SSD, Caffe model) locates the face, then the upper-middle region of the face (where eyes are) is cropped and classified by your trained CNN
- **Deployment:** Streamlit web app with live camera, snapshot, and upload-based testing
