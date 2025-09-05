import jinja2

def inspect_prompt(prompt_content, model_name, tokenizer=None):
    """
    Applies model-specific formatting to a prompt and prints it for inspection.

    Args:
        prompt_content (str): The core text of the prompt (e.g., from a Jinja template).
        model_name (str): A string identifier for the model (e.g., 'gpt-oss-20b').
        tokenizer (AutoTokenizer, optional): The tokenizer for the model. 
                                            Highly recommended for standard models.

    Returns:
        str: The fully formatted, model-ready prompt string.
    """
    
    formatted_prompt = ""
    formatting_method = "Unknown"

    # --- Logic to Apply Formatting ---
    
    # 1. Best Practice: Try using the tokenizer's chat template if available
    if tokenizer and tokenizer.chat_template:
        formatting_method = "Tokenizer Chat Template"
        # The apply_chat_template function expects a list of dictionaries (a conversation)
        messages = [{"role": "user", "content": prompt_content}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 2. Manual Fallbacks for models with known custom formats
    else:
        if "gpt-oss-20b" in model_name:
            formatting_method = "Manual Format for gpt-oss-20b"
            formatted_prompt = f"<|channel|>user<|message|>{prompt_content}<|channel|>assistant<|message|>"
        
        elif "Llama-3" in model_name:
            formatting_method = "Manual Format for Llama-3"
            # Note: This is a simplified example. Using the tokenizer is always better for Llama.
            formatted_prompt = f"<s>[INST] {prompt_content} [/INST]"
            
        # 3. Default: If no other format matches, just use the raw content
        else:
            formatting_method = "Default (No Formatting)"
            formatted_prompt = prompt_content

    # --- Print for Manual Inspection ---
    
    print("=" * 60)
    print(f"INSPECTING PROMPT FOR MODEL: '{model_name}'")
    print(f"(Using Method: {formatting_method})")
    print("-" * 60)
    print(formatted_prompt)
    print("=" * 60)
    
    return formatted_prompt