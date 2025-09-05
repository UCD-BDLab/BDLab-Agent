# import torch
# import re
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import numpy as np
# import faiss
# #from backend.tools.retrieval import retrieve_context, raw_chunks
# from .embed import chunk_files, embed_chunks

# def chat_oss(tokenizer, model, user_prompt, max_new_tokens=256, do_sample=False, temperature=0.0):
#     """
#     This function builds the prompt manually using the model's native format.
#     """
#     # Manually construct the prompt using the newly discovered special tokens
#     # We prompt the 'assistant' channel to give us the final answer.
#     prompt = f"<|channel|>user<|message|>{user_prompt}<|channel|>assistant<|message|>"

#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.inference_mode():
#         out = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens, # Increased to give model space to answer
#             do_sample=do_sample,
#             temperature=temperature,
#             # We don't know the exact EOS token, so we rely on max_new_tokens for now.
#             # The model will likely generate a <|channel|> or other special token to stop.
#             pad_token_id=tokenizer.pad_token_id,
#         )
#     gen_ids = out[0, inputs["input_ids"].shape[-1]:]
#     return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


# # # --- TESTING THE NEW FUNCTION ---
# # user_question = "Explain Harmony format in one sentence."
# # response = chat_oss_corrected(user_question)
# # print(response)

# def ask_rag(query):
#     """
#     Performs the full RAG pipeline: retrieve, prompt, generate, and cleanup.
#     """
#     # 1. Retrieve context
#     retrieved_context = retrieve_context(query, k=3)

#     # 2. Create the RAG prompt (this part is the same)
#     prompt_template = """
#         You are a helpful assistant for answering questions about US Senate bills.
#         Use the following context to answer the user's question.
#         If the answer is not found in the context, state that you cannot find the answer in the provided documents.
#         Do not use any external knowledge or make up information.

#         --- CONTEXT ---
#         {context}
#         --- END CONTEXT ---

#         USER'S QUESTION: {question}
#     """.strip()

#     final_prompt_text = prompt_template.format(context=retrieved_context, question=query)

#     print("\n--- GENERATING RESPONSE ---")
#     # 3. Generate the raw response
#     raw_response = chat_oss(final_prompt_text, max_new_tokens=512)

#     # cleaning up the response
#     # We split the response by our keyword and take the last part.
#     # This removes the "chain-of-thought" text.
#     parts = raw_response.split("assistantfinal")
#     clean_response = parts[-1].strip()

#     return clean_response


# def retrieve_context(embedder,raw_chunks, index, query, k=5):
#     """
#     Retrieves the top-k most relevant chunks from the FAISS index for a given query.
#     """
#     print(f"Retrieving context for query: '{query}'")

#     # 1. Embed the query
#     query_emb = embedder.encode([query], convert_to_numpy=True).astype(np.float32)

#     # 2. Normalize the query embedding (important for cosine similarity with IndexFlatIP)
#     faiss.normalize_L2(query_emb)

#     # search the FAISS index for top k
#     distances, indices = index.search(query_emb, k)

#     # Fetching text chunks using the indices
#     retrieved_chunks= []
#     for i in indices[0]:
#         retrieved_chunks.append(raw_chunks[i])

#     # 5. Combine the chunks into a single context string
#     context = "\n\n---\n\n".join(retrieved_chunks)

#     print("Context retrieved successfully.")
#     return context