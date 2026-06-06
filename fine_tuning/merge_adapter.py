"""Merge LoRA adapter weights into base model for Ollama import."""
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

ADAPTER_PATH = "./lora-clinical/final"
MERGED_PATH  = "./lora-clinical/merged"

model = AutoPeftModelForCausalLM.from_pretrained(ADAPTER_PATH)
merged = model.merge_and_unload()
merged.save_pretrained(MERGED_PATH)
AutoTokenizer.from_pretrained(ADAPTER_PATH).save_pretrained(MERGED_PATH)
print(f"✓ Merged model saved → {MERGED_PATH}")
print("Convert to GGUF for Ollama: llama.cpp convert-hf-to-gguf.py " + MERGED_PATH)
