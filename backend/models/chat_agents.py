import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_llm(base_path):
    # Load the model and tokenizer from local path
    base = Path("/home/vicente/Github/BDLab-Agent/backend/data/GPTModels/gpt-oss-20b")
    model = AutoModelForCausalLM.from_pretrained(str(base),dtype=torch.bfloat16,device_map="auto",local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(base),local_files_only=True)
    model.eval()
    return model, tokenizer

def chat_oss(model, tokenizer, user_prompt, system_prompt=None, max_new_tokens=512, do_sample=True, temperature=0.1):
    """
    Core function to generate a response from the LLM without external context.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    prompt_length = inputs["input_ids"].shape[1]
    raw_output = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    
    clean_response = raw_output.split("assistantfinal")[-1].strip()
    if clean_response.startswith("analysis"):
        clean_response = clean_response[len("analysis"):].strip()
        
    return clean_response