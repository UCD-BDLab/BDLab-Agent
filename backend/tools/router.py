# def route_query(user_question: str) -> str:
#     """Decides whether to search the tree or the vector store."""
    
#     prompt = f"""
#     You are a helpful routing assistant. Based on the user's question, should I search the 'KNOWLEDGE_TREE' or the 'VECTOR_DATABASE'?
    
#     - Use 'KNOWLEDGE_TREE' for specific questions about code names, definitions, and categories.
#     - Use 'VECTOR_DATABASE' for broader, conceptual, or explanatory questions.
    
#     User question: "{user_question}"
    
#     Optimal data source:
#     """
    
#     # The LLM will respond with either "KNOWLEDGE_TREE" or "VECTOR_DATABASE"
#     decision = call_llm(prompt).strip()
    
#     if "KNOWLEDGE_TREE" in decision:
#         return "tree"
#     else:
#         return "vector"