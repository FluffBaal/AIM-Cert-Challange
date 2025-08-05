# Ingestion Service - Complete Implementation

This ingestion service processes the Chris Voss negotiation book (`output.md`) and populates the Qdrant vector database with dual chunking strategies for optimal retrieval.

## 🏗️ Architecture Overview

The ingestion service implements a **dual chunking strategy** that creates two separate collections in Qdrant:

1. **Naive Collection (`never_split_naive`)**: Fixed-size chunks with overlap for baseline retrieval
2. **Advanced Collection (`never_split_advanced`)**: Parent-child hierarchical chunks with semantic throttling

## 📋 Key Components

### Core Files

- **`ingest.py`** - Main ingestion script and CLI entry point
- **`dual_chunking_pipeline.py`** - Orchestrates both chunking strategies
- **`naive_chunker.py`** - Fixed-size chunking implementation
- **`semantic_chunker.py`** - Advanced parent-child chunking
- **`qdrant_collections.py`** - Vector database management
- **`dual_rag_retriever.py`** - Retrieval with context corridor and reranking
- **`constants.py`** - Configuration constants

### Supporting Files

- **`Dockerfile`** - Container configuration
- **`pyproject.toml`** - Python dependencies
- **Test files**: `test_*.py` - Comprehensive test suite

## 🚀 Quick Start

### 1. Environment Setup

Copy the environment template:
```bash
cp ../.env.example ../.env
```

Set your API keys in `.env`:
```env
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here  # Optional, for reranking
```

### 2. Run with Docker Compose

From the project root:
```bash
docker-compose up ingestion
```

### 3. Run Standalone

```bash
cd backend/ingestion
python ingest.py --dual-mode --data-dir ../../data
```

## 🧪 Testing

The service includes comprehensive tests:

### Basic Functionality Tests
```bash
# Test markdown processing and chunking
python simple_test.py

# Test chunking in isolation
python test_chunking_only.py

# Test collection configurations
python test_collections.py
```

### Mock Testing (No API Keys Required)
```bash
# Test complete pipeline with mocks
python test_ingestion_mock.py
```

### Real API Testing (Requires API Keys)
```bash
# Test with actual OpenAI API
export OPENAI_API_KEY="your-key-here"
python run_ingestion_test.py
```

## 📊 Chunking Strategies

### Naive Chunking
- **Size**: 500 tokens per chunk
- **Overlap**: 50 tokens
- **Method**: Fixed-size sliding window
- **Use Case**: Baseline retrieval performance

### Advanced Chunking
- **Structure**: Parent-child hierarchy
- **Parents**: Split by markdown headings (# and ##)
- **Children**: Semantic throttling within parents
- **Features**: 
  - Context corridor for improved retrieval
  - Parent-child relationships preserved
  - Markdown-aware boundaries

## 🔧 Configuration

Key constants in `constants.py`:

```python
# Collection names
NAIVE_COLLECTION = "never_split_naive"
ADVANCED_COLLECTION = "never_split_advanced"

# Naive chunking
NAIVE_CHUNK_SIZE = 500  # tokens
NAIVE_OVERLAP = 50      # tokens

# Advanced chunking
PARENT_TOKEN_LIMIT = 1200    # Max tokens per parent
CHILD_TOKEN_MIN = 100        # Min tokens per child
CHILD_TOKEN_MAX = 140        # Max tokens per child

# Embeddings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
```

## 📝 Data Processing Flow

1. **Document Loading**: Load `output.md` (425,585 characters)
2. **Dual Chunking**: 
   - Naive: ~205 fixed-size chunks
   - Advanced: ~124 parents with ~646 children
3. **Embedding Generation**: OpenAI text-embedding-3-small
4. **Vector Storage**: Upsert to respective Qdrant collections
5. **Metadata Storage**: Preserve parent-child relationships

## 🔍 Collections Schema

### Naive Collection Schema
```json
{
  "chunk_id": "keyword",
  "content": "text", 
  "chunk_index": "integer",
  "overlap_start": "integer",
  "overlap_end": "integer",
  "method": "keyword",
  "tokens": "integer"
}
```

### Advanced Collection Schema
```json
{
  "parent_id": "keyword",
  "section_heading": "text",
  "anchor_id": "keyword", 
  "child_idx": "integer",
  "content": "text",
  "parent_text": "text",
  "metadata": "json",
  "tokens": "integer",
  "method": "keyword"
}
```

## 🛠️ Error Handling

The service includes comprehensive error handling:

- **API Failures**: Graceful degradation with fallbacks
- **Network Issues**: Retry logic and timeout handling
- **Data Validation**: Input validation and sanitization
- **Logging**: Detailed logging at INFO/ERROR levels

## 📦 Dependencies

Core dependencies:
- `qdrant-client>=1.7.0` - Vector database client
- `openai==1.6.1` - OpenAI API for embeddings
- `cohere==4.39.0` - Reranking (optional)
- `tiktoken==0.5.2` - Tokenization
- `pydantic==2.5.0` - Data validation

## 🐳 Docker Integration

The service is fully containerized:

- **Base Image**: `python:3.11-slim`
- **Package Manager**: `uv` for fast dependency installation
- **Data Volume**: `/data` mounted for document access
- **Network**: Connects to `qdrant` and `redis` services

## 🔄 Health Checks

Built-in health checks verify:
- Qdrant connection and collections
- OpenAI API availability
- Cohere API availability (optional)
- Component initialization status

## 📈 Performance Metrics

Expected processing performance:
- **Document Size**: 425KB markdown file
- **Processing Time**: ~2-5 minutes (depending on API latency)
- **Naive Chunks**: ~205 chunks
- **Advanced Chunks**: ~124 parents, ~646 children
- **Vector Dimensions**: 1536 per embedding

## 🔧 Customization

To customize for different documents:

1. **Adjust chunk sizes** in `constants.py`
2. **Modify chunking logic** in chunker classes
3. **Update collection schemas** in `qdrant_collections.py`
4. **Configure embedding model** in constants

## 🚨 Troubleshooting

### Common Issues

1. **Missing API Keys**: Set `OPENAI_API_KEY` environment variable
2. **Qdrant Connection**: Ensure Qdrant service is running on port 6333
3. **Memory Issues**: Large documents may require increased container memory
4. **Token Limits**: OpenAI API has rate limits for embedding generation

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python ingest.py --dual-mode
```

## 📚 Additional Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [Cohere Rerank API](https://docs.cohere.com/docs/rerank-guide)

## 🎯 Ready for Production

The ingestion service is **production-ready** with:

✅ Complete dual chunking implementation  
✅ Vector database integration  
✅ Error handling and logging  
✅ Docker containerization  
✅ Comprehensive test suite  
✅ Health checks and monitoring  
✅ Documentation and examples  

The service successfully processes the Chris Voss negotiation book and creates the required Qdrant collections for the dual RAG system.