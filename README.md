A Generative AI–powered medical report summarization system that converts scanned medical reports (PNG images) into clean, patient-friendly summaries. Uses OCR (Tesseract) to extract text from medical images and Flan-T5 to generate concise summaries. Supports text cleaning, medical term simplification, and Streamlit UI.
Features

Accepts PNG medical reports
Extracts text using Tesseract OCR
Summarizes content using Flan-T5 (open-source LLM)
Optional patient-friendly summary version
Streamlit interface for easy use
Runs locally on CPU
Modular design: OCR → Text → Summary
