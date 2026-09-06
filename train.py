import os
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

# 👇 Import your data pipeline from the other file
from data_prep import prepare_dataset

def main():
    # 1. CONFIGURATION
    MODEL_ID = "Qwen/Qwen2.5-1.5B"
    PDF_PATH = "intro-to-ml.pdf"  # 👈 Put the PDF in the same folder as this script
    OUTPUT_DIR = "./results"
    BLOCK_SIZE = 1024             # 👈 Matches data_prep.py
    LEARNING_RATE = 5e-5          # 👈 Matches SFTConfig below

    # 2. LOAD DATA
    print("Loading and preparing dataset...")
    dataset, tokenizer = prepare_dataset(PDF_PATH, block_size=BLOCK_SIZE)

    # Split 90/10 for validation
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train blocks: {len(train_ds)} | Eval blocks: {len(eval_ds)}")

    # 3. LOAD MODEL
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    # Safe token loading: uses env var if present, otherwise None (public model)
    hf_token = os.environ.get("HF_TOKEN")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        token=hf_token,
    )
    model.config.use_cache = False

    # 4. SETUP LORA
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=32, lora_alpha=64, lora_dropout=0.1,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 5. TRAINING ARGS
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_seq_length=BLOCK_SIZE,
        num_train_epochs=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="adamw_torch",
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
    )

    # 6. TRAIN
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    steps_per_epoch = len(train_ds) // 8
    print(f"Steps per epoch: {steps_per_epoch} | Total steps: {steps_per_epoch * 4}")
    
    trainer.train()
    
    # Save final
