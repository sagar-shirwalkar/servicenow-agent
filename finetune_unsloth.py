from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

#
# Unsloth makes training 2x faster and uses 80% less memory. You will run this script, which will output a GGUF file.
# Note: The exact Unsloth boilerplate is extensive, but here is the core logic you will adapt from their official GitHub repo. 
#

# Load base model (Qwen 2.5 Coder 7B)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-Coder-7B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model, r = 16, target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16, lora_dropout = 0, bias = "none", use_gradient_checkpointing = "unsloth"
)

# Load dataset and train
dataset = load_dataset("json", data_files="train_data.json", split="train")
trainer = SFTTrainer(model=model, train_dataset=dataset, dataset_text_field="text", max_seq_length=2048)
trainer.train()

# Export to GGUF for Ollama
model.save_pretrained_gguf("qwen_servicenow_gguf", tokenizer, quantization_method = "q6_k")
