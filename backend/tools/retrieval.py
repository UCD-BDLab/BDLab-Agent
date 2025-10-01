import faiss
import numpy as np
import sys
from backend.models.chat_agents import chat_oss

def retrieve_context(query, k=3, embedder=None, index=None, raw_chunks=None, chunk_metadata=None):
    """
    Retrieves the top-k most relevant chunks from the FAISS index for a given query.
    """
    print(f"Retrieving context for query: '{query}'")

    # embed the query
    query_emb = embedder.encode([query], convert_to_numpy=True).astype(np.float32)

    # normalize the query embedding (for cosine similarity)
    faiss.normalize_L2(query_emb)

    # search the FAISS index
    distances, indices = index.search(query_emb, k)

    # fetch the actual text chunks using the indices
    retrieve_chunks_text = []
    for i in indices[0]:
        retrieve_chunks_text.append(raw_chunks[i])

    retrieved_chunks_meta = []
    for i in indices[0]:
        retrieved_chunks_meta.append(chunk_metadata[i])

    # we combine single context string
    context = "\n\n---\n\n".join(retrieve_chunks_text)

    print("Context retrieved successfully.")

    # Return both the context string AND the list of metadata dictionaries
    return context, retrieved_chunks_meta


def ask_rag(query, model=None, tokenizer=None, embedder=None, index=None, raw_chunks=None, chunk_metadata=None):
    """
    The complete RAG pipeline.
    Retrieves context, builds a prompt, and generates an answer with sources.
    """
    # First retrieve context
    retrieved_context, sources = retrieve_context(query, k=3)

    # Now we create the RAG prompt:
    # This Combine the context and query into a single prompt for the LLM, (instructing it on how to behave)
    augmented_prompt  = """
        You are a helpful assistant for answering questions about US Senate bills and Acts.
        Use the following context to answer the user's question.
        If the answer is not found in the context, state that you cannot find the answer in the provided documents.
        Do not use any external knowledge or make up information.

        (START CONTEXT): {context} (END CONTEXT).

        USER QUESTION: {question} """.strip()

    # based on the prompt template, we create the final prompt text passing in the retrieved context and user question
    final_prompt_text = augmented_prompt.format(context=retrieved_context, question=query)

    print("\nGENERATING RESPONSE:\n")

    # passing the fully formatted RAG prompt as the "user_prompt"
    response = chat_oss(final_prompt_text, max_new_tokens=512, do_sample=False)

    # print("SOURCES USED:\n")
    # for i, meta in enumerate(sources):
    #     print(f"Source {i+1}: {meta['title']} (Chunk ID: {meta['chunk_id']})")
    
    return response, sources