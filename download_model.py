from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HF_MODEL_NAME = "williamhtan/nllb-200-distilled-600M_dyu-fra"
OUTPUT_DIR = "./saved_model"

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, src_lang="dyu_Latn", tgt_lang="fr_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(HF_MODEL_NAME)

tokenizer.save_pretrained(OUTPUT_DIR)
model.save_pretrained(OUTPUT_DIR)
