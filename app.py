import os
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Skin Cancer Image Analysis", page_icon="🔬", layout="wide")

st.title("🔬 Skin Cancer Image Analysis")
st.caption("Dataset explorer for the Skin Cancer Malignant vs Benign dataset")
st.warning(
    "Research/education demo only. This app does not diagnose skin cancer. "
    "The supplied Colab notebooks show dataset downloading and image exploration, "
    "but do not contain a trained classification model. No benign/malignant prediction is made."
)

DATA_DIR = Path("skin_images")
SPLITS = ("train", "test")
CLASSES = ("benign", "malignant")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def image_files(folder: Path):
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


st.sidebar.header("Dataset")
st.sidebar.write(f"Folder: `{DATA_DIR}`")

if not DATA_DIR.exists():
    st.info(
        "Dataset folder not found. Add the extracted `skin_images` folder to the GitHub repository. "
        "It should contain `train/benign`, `train/malignant`, `test/benign`, and `test/malignant`. "
        "Alternatively, upload images below to preview them."
    )
else:
    records = []
    for split in SPLITS:
        for label in CLASSES:
            folder = DATA_DIR / split / label
            files = image_files(folder)
            records.append({"Split": split.title(), "Class": label.title(), "Images": len(files)})

    df = pd.DataFrame(records)
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Training images", int(df.loc[df.Split == "Train", "Images"].sum()))
    c2.metric("Testing images", int(df.loc[df.Split == "Test", "Images"].sum()))
    c3.metric("Total images", int(df["Images"].sum()))
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Class distribution")
    chart_df = df.pivot(index="Class", columns="Split", values="Images")
    st.bar_chart(chart_df)

    st.subheader("Sample dataset images")
    split_choice = st.selectbox("Choose dataset split", SPLITS)
    class_choice = st.selectbox("Choose image class", CLASSES)
    samples = image_files(DATA_DIR / split_choice / class_choice)
    if samples:
        cols = st.columns(3)
        for i, img_path in enumerate(samples[:9]):
            try:
                with Image.open(img_path) as img:
                    cols[i % 3].image(img.copy(), caption=img_path.name, use_container_width=True)
            except Exception:
                cols[i % 3].write(f"Could not open: {img_path.name}")
    else:
        st.warning("No images found in this class folder.")

st.divider()
st.subheader("Upload an image to preview")
uploaded = st.file_uploader("Choose a skin image", type=["jpg", "jpeg", "png", "webp", "bmp"])
if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", width=420)
        st.info(
            "Image preview only. A trained model file and validated inference pipeline are not present "
            "in the provided notebooks, so this app cannot classify the image."
        )
    except Exception as exc:
        st.error(f"Unable to open this image: {exc}")

st.caption("For any concerning or changing skin lesion, seek assessment from a qualified healthcare professional.")
