import streamlit as st
from PIL import Image
import pytesseract
from summarizer import MedicalSummarizer

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.title("🩺 Medical Report Summarizer (PNG Supported)")

uploaded_file = st.file_uploader("Upload Medical Report PNG", type=["png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Report", use_column_width=True)

    extracted_text = pytesseract.image_to_string(img)

    st.subheader("Extracted Text (OCR):")
    st.text(extracted_text)

    summarizer = MedicalSummarizer()

    if st.button("Generate Summary"):
        summary = summarizer.summarize(extracted_text)
        st.subheader("Summary:")
        st.success(summary)
