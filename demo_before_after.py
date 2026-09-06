import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

def generate(pipe, prompt, max_new_tokens=60):
    """Helper function to generate text with consistent settings."""
    output = pipe(
        prompt, 
        max_new_tokens=max_new_tokens, 
        do_sample=True, 
        temperature=0.7, 
        top_p=0.9,
        pad_token_id=pipe.tokenizer.eos_token_id
    )
    return output[0]["generated_text"]

def main():
    BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B"
    # 👈 Update this to wherever your adapter is saved locally
    ADAPTER_DIR = "./results" 
    
    print("1. Loading base Qwen model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, 
        torch_dtype=torch.float16, 
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    print(f"2. Attaching LoRA adapter from {ADAPTER_DIR}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    
    print("3. Initializing pipeline...")
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        device_map="auto"
    )
    
    prompt = "In supervised learning, the training data consists of"
    print(f"\nPrompt: '{prompt}'\n")
    print("=" * 60)
    
    # === BEFORE: Pure Base Model (Adapter OFF) ===
    print("🔴 BEFORE: Pure Base Model (Adapter OFF)")
    print("   (The model hasn't learned the book yet)")
    print("-" * 60)
    with model.disable_adapter():
        base_output = generate(pipe, prompt)
        print(base_output)
        
    print("\n" + "=" * 60 + "\n")
    
    # === AFTER: Fine-Tuned Model (Adapter ON) ===
    print("🟢 AFTER: Fine-Tuned Model (Adapter ON)")
    print("   (The model uses the knowledge from the ML book)")
    print("-" * 60)
    ft_output = generate(pipe, prompt)
    print(ft_output)

if __name__ == "__main__":
    main()
