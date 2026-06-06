"""
LoRA fine-tuning script — designed to run on Colab T4 (free tier).
Upload this file + clinical_qa.jsonl to Colab, then run.

Colab setup:
  !pip install transformers peft trl datasets bitsandbytes accelerate -q
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_DIR = "./lora-clinical"

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
)

dataset = load_dataset("json", data_files="clinical_qa.jsonl", split="train")

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, load_in_4bit=True, device_map="auto")
model = get_peft_model(model, lora_config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=1024,
)

print("Starting LoRA fine-tuning…")
trainer.train()
trainer.model.save_pretrained(f"{OUTPUT_DIR}/final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print(f"✓ LoRA adapter saved → {OUTPUT_DIR}/final")
print("Download and merge with: python fine_tuning/merge_adapter.py")
