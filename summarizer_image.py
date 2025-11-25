import pytesseract
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- 1. OCR SETUP ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  
# Change the path if needed

def extract_text_from_png(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text


# --- 2. SUMMARIZER MODEL ---
class MedicalSummarizer:
    def __init__(self, model_name="google/flan-t5-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(self, text, max_len=200):
        inputs = self.tokenizer(
            "Summarize this medical report: " + text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_len,
            min_length=40,
            num_beams=4,
            temperature=0.7
        )

        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# --- 3. COMPLETE PIPELINE ---
def summarize_medical_png(image_path):
    extracted_text = extract_text_from_png(image_path)

    print("\n--- Extracted OCR Text ---\n")
    print(extracted_text)

    summarizer = MedicalSummarizer()
    summary = summarizer.summarize(extracted_text)

    print("\n--- Summary ---\n")
    print(summary)
    return summary
