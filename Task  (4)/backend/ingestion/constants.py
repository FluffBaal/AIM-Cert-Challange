"""
Chunking Constants from Task 3 - Section 5.2 (lines 1999-2002 reference)

These constants define the parameters for both naive and advanced chunking strategies,
as well as retrieval thresholds for the dual RAG system.
"""

# Chunking Constants from Task 3
SIMILARITY_MERGE_THRESHOLD = 0.92  # For semantic throttling
SIMILARITY_SPLIT_THRESHOLD = 0.15  # For re-splitting
PARENT_TOKEN_LIMIT = 1200         # Max tokens per parent
CHILD_TOKEN_MIN = 100            # Min tokens per child  
CHILD_TOKEN_MAX = 140            # Max tokens per child
THIN_PARENT_MIN = 400            # Min tokens for parent
CONTEXT_CORRIDOR_THRESHOLD = 0.55 # For including preceding parent

# Collection names
NAIVE_COLLECTION = "never_split_naive"
ADVANCED_COLLECTION = "never_split_advanced"

# Naive chunking settings
NAIVE_CHUNK_SIZE = 500  # tokens
NAIVE_OVERLAP = 50      # tokens

# Vector embedding settings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Reranking settings
COHERE_MODEL = "rerank-v3.5"  # Latest model as of 2025
NAIVE_RETRIEVAL_LIMIT = 20  # Retrieve more candidates for reranking
ADVANCED_RETRIEVAL_LIMIT = 30  # Get more parents for reranking (increased for better coverage)
FINAL_RESULTS_LIMIT = 5  # Return top 5 after reranking (naive)
ADVANCED_RESULTS_LIMIT = 7  # Top 7 reranked for advanced (increased for more context)