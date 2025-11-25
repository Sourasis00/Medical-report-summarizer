from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class MedicalSummarizer:
    def __init__(self, model_name="google/flan-t5-large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(self, text, max_len=150):
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
