import os
import re
import pymupdf
from transformers import AutoTokenizer
from datasets import Dataset

def prepare_dataset(pdf_path: str, block_size: int = 1024) -> Dataset:
    """
    Extracts text from a PDF, cleans it, and tokenizes it into 
    fixed-size blocks for continued pretraining.
    """
    print(f"[1/4] Extracting text from {pdf_path}...")
    doc = pymupdf.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    
    print("[2/4] Cleaning OCR artifacts and page numbers...")
    # Specific fixes for the Intro to ML book
    text = text.replace('1n[', 'In[')
    text = text.replace('mgllearn', 'mglearn')
    text = text.replace('0ut[', 'Out[')
    text = re.sub(r'\n\d+\n', '\n', text) 
    
    print("[3/4] Tokenizing paragraphs and packing into blocks...")
    # Qwen is public. We use token=False, but allow HF_TOKEN if it exists in the environment.
    hf_token = os.environ.get("HF_TOKEN") 
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", token=hf_token)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Rebuild logical paragraphs (filter garbage < 100 chars)
    paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) >= 100]
    
    # Tokenize + append EOS token to mark paragraph boundaries
    token_lists = tokenizer(paras, add_special_tokens=False)["input_ids"]
    all_ids = []
    for ids in token_lists:
        all_ids.extend(ids + [tokenizer.eos_token_id])
        
    # Pack into exact blocks (drops remainder)
    n_blocks = len(all_ids) // block_size
    input_ids = [all_ids[i*block_size : (i+1)*block_size] for i in range(n_blocks)]
    
    dataset = Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": [[1] * block_size] * n_blocks,
    })
    
    print(f"[4/4] Shuffling dataset...")
    dataset = dataset.shuffle(seed=42)
    
    print(f"✅ Success! Created {len(dataset)} blocks of {block_size} tokens.")
    return dataset, tokenizer


if __name__ == "__main__":
    # 👇 INTERVIEWER NOTE: Change this path to run locally!
    PDF_FILE = "intro-to-ml.pdf"  
    
    dataset, tokenizer = prepare_dataset(PDF_FILE, block_size=1024)
    
    print("\n--- Preview of first training block ---")
    print(tokenizer.decode(dataset[0]["input_ids"])[:300])
