# Freelancer Negotiation Helper

A real-time negotiation assistance tool that helps freelancers defend their worth during price negotiations using Chris Voss's techniques from "Never Split the Difference".

## Project Structure

```
.
├── docker-compose.yml           # Docker orchestration for all services
├── backend/
│   ├── api/                    # FastAPI backend
│   │   ├── main.py            # API entry point with middleware
│   │   ├── dependencies.py     # Service dependencies
│   │   ├── requirements.txt    # Python dependencies
│   │   ├── Dockerfile         # API container config
│   │   └── routers/           # API endpoints
│   │       ├── negotiation.py # Negotiation endpoints
│   │       └── websocket.py   # WebSocket support
│   ├── agents/                # Multi-agent system
│   │   ├── models.py          # Data models and config
│   │   ├── supervisor_agent.py # Main orchestrator
│   │   ├── web_search_agent.py # Market data search
│   │   ├── rag_search_agent.py # Vector search
│   │   ├── synthesis_agent.py  # Response synthesis
│   │   └── error_handler.py   # Error handling & circuit breaker
│   ├── ingestion/             # Data pipeline
│   │   ├── constants.py       # Chunking constants
│   │   ├── naive_chunker.py   # Simple chunking
│   │   ├── semantic_chunker.py # Advanced chunking
│   │   ├── qdrant_collections.py # Vector DB setup
│   │   ├── dual_rag_retriever.py # Dual retrieval
│   │   └── ingest.py          # Main ingestion script
│   └── evaluation/            # Ragas evaluation
│       ├── requirements.txt   # Evaluation dependencies
│       └── Dockerfile         # Evaluation container
├── frontend/                  # React + TypeScript frontend
│   ├── src/
│   │   ├── components/       # UI components
│   │   │   ├── ui/          # shadcn/ui components
│   │   │   └── settings/    # Settings page
│   │   ├── services/        # API client
│   │   ├── store/           # Zustand state management
│   │   └── types/           # TypeScript types
│   ├── package.json         # Frontend dependencies
│   ├── Dockerfile           # Frontend container
│   └── nginx.conf           # Production server config
└── data/                    # Document storage
    └── (Place PDF files here)
```

## Key Features

### Backend Architecture
- **Multi-Agent System**: Supervisor orchestrates parallel execution of specialized agents
- **Dual RAG Strategy**: Naive vs Advanced chunking with comparison metrics
- **Self-Correcting Search**: Web search with quality grading and iteration
- **Comprehensive Error Handling**: Circuit breakers, rate limiting, and graceful degradation
- **Real-time Support**: WebSocket connections for live negotiation assistance

### Frontend Features
- **API Key Management**: Secure local storage with validation
- **Comparison View**: Side-by-side naive vs advanced RAG results
- **Real-time Updates**: WebSocket integration for instant feedback
- **Performance Metrics**: Detailed analysis of retrieval strategies

### Model Configuration
```python
MODEL_CONFIG = {
    "supervisor": "gpt-4.1",      # Advanced reasoning
    "synthesis": "gpt-4.1",       # High-quality writing
    "web_search": "gpt-4.1-mini", # Fast search
    "rag_search": "gpt-4.1-mini", # Fast retrieval
}
```

## Getting Started

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ with [uv](https://github.com/astral-sh/uv) installed
- Node.js 18+ and npm
- API Keys:
  - OpenAI API Key (required)
  - Cohere API Key (required for reranking)
  - Exa API Key (required for web search)
  - LangSmith API Key (optional for monitoring)

### Quick Start

1. Clone the repository
2. Add your PDF documents to the `data/` directory
3. Create `.env` file with system API keys:
   ```
   OPENAI_API_KEY=your_key_here
   LANGSMITH_API_KEY=your_key_here  # Optional
   ```
4. Start all services:
   ```bash
   docker-compose up -d
   ```
5. Access the application at http://localhost:3000

### Data Ingestion

To ingest documents into the vector database:
```bash
docker-compose run ingestion
```

This will process all PDFs in the `data/` directory using both naive and advanced chunking strategies.

## API Endpoints

- `POST /api/negotiate` - Analyze negotiation text
- `POST /api/negotiate/compare` - Compare naive vs advanced RAG
- `GET /api/test-api-keys` - Validate API keys
- `GET /health` - Service health check
- `WS /ws/negotiate` - WebSocket for real-time analysis

## Implementation Notes

This implementation follows the exact specifications from the build plan, including:
- Standardized model configuration (GPT-4.1 family)
- Dual Qdrant collections for comparison
- Context corridor logic for advanced retrieval
- Cohere reranking for improved relevance
- Comprehensive fallback strategies
- Security-first API key handling

## Development

### Setup Development Environment
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
make install-dev

# Or manually for each service
cd backend/api && uv pip sync
cd backend/ingestion && uv pip sync
cd backend/evaluation && uv pip sync
```

### Run Services in Development Mode
```bash
# Using Make commands
make dev-api        # Start API server
make dev-frontend   # Start frontend dev server

# Or manually
cd backend/api && uv run uvicorn main:app --reload
cd frontend && npm install && npm run dev
```

### Code Quality
```bash
# Format code
make format

# Run linters
make lint

# Type checking
make type-check

# Run all tests
make test
```

### Docker Development
```bash
# Build all containers
make docker-build

# Start all services
make docker-up

# View logs
make docker-logs

# Stop all services
make docker-down
```

## License

This project implements the architecture specified in the comprehensive build plan for the Freelancer Negotiation Helper.