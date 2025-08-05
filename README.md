# AIM Certificate Challenge - Advanced RAG Implementation

## 🎯 Project Overview

This repository contains my submission for the AIM Certificate Challenge, featuring an innovative Parent-Child Retrieval Architecture for RAG (Retrieval-Augmented Generation) systems. The project challenges traditional evaluation metrics and embraces the future of large context windows in modern LLMs.

### 🔑 Key Innovation

**Parent-Child Chunking Strategy**: A hierarchical approach to document chunking that maintains context relationships, designed for the era of 128k+ token context windows.

## 📁 Repository Structure

```
AIM Cert Challenge/
├── Task  (1)/          # Initial exploration and setup
├── Task  (2)/          # System design and architecture
├── Task  (3)/          # Implementation planning
├── Task  (4)/          # Main RAG implementation
│   ├── backend/        # FastAPI services
│   ├── frontend/       # React UI
│   └── data/           # Chris Voss negotiation content
├── Task  (5)/          # Performance analysis
├── Task  (6)/          # Retrieval techniques comparison
├── Task  (7)/          # Future directions
└── Final Submission/   # Project index and overview
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose
- OpenAI API key
- Cohere API key (optional, for reranking)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone [your-repo-url]
   cd "AIM Cert Challenge"
   ```

2. **Navigate to the main implementation**
   ```bash
   cd "Task  (4)"
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   # OPENAI_API_KEY=your-key-here
   # COHERE_API_KEY=your-key-here (optional)
   ```

4. **Run with Docker Compose**
   ```bash
   docker-compose up
   ```

   Or run services individually:
   ```bash
   # Backend API
   cd backend/api
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python main.py

   # Frontend
   cd frontend
   npm install
   npm run dev
   ```

## 📊 Key Findings

### Performance Metrics

| Retrieval Method | RAGAS Score | Real-World Performance |
|-----------------|-------------|------------------------|
| Naive Baseline | 0.743 | Good precision, limited context |
| Parent-Child | 0.435 | Poor precision, excellent context |
| Ensemble | 0.798 | Best overall balance |

### The Paradox

While Parent-Child scored poorly on RAGAS metrics, it provides superior context for generation tasks. This highlights a fundamental mismatch between current evaluation metrics and the capabilities of modern LLMs.

## 🏗️ Architecture Highlights

### Dual Chunking Pipeline
- **Naive Strategy**: Fixed-size chunks (500 tokens) with overlap
- **Advanced Strategy**: Hierarchical parent-child relationships
  - Parents: 600-1200 tokens (document sections)
  - Children: 100-140 tokens (searchable units)

### Context Corridor
An innovative feature that retrieves adjacent document sections when high similarity is detected, maintaining narrative flow.

### Technology Stack
- **Backend**: FastAPI, Qdrant Vector DB, Redis
- **Frontend**: React, TypeScript, Tailwind CSS
- **ML/AI**: OpenAI Embeddings, Cohere Reranking
- **Infrastructure**: Docker, Docker Compose

## 📚 Documentation

For detailed documentation, see:
- [Project Overview and Index](Final%20Submission/Final.md)
- [Implementation Details](Task%20%20(4)/README.md)
- [Parent-Child Architecture Guide](Task%20%20(4)/PARENT_CHILD_CHUNKING_IMPLEMENTATION_GUIDE.md)
- [Performance Analysis](Task%20%20(5)/Task_05.md)

## 🎓 Key Learnings

1. **Metrics vs Reality**: RAGAS optimizes for precision in a world moving toward comprehensive context
2. **Simple Often Wins**: Ensemble methods outperformed complex architectures
3. **Future-Focused Design**: Building for tomorrow's LLMs, not yesterday's constraints
4. **Context Matters**: Users prefer comprehensive answers over precise snippets

## 🔮 Future Vision

This project represents a bet on the future of RAG systems where:
- LLMs have million-token contexts
- Retrieval provides rich, interconnected knowledge graphs
- Systems optimize for understanding, not just precision
- Evaluation metrics evolve to match capabilities

## 📄 License

This project is submitted as part of the AIM Certificate Challenge.

## 🙏 Acknowledgments

- Chris Voss for "Never Split the Difference" - the foundation of our negotiation content
- The AIM Certificate program for this challenging and educational project
- The open-source community for the amazing tools that made this possible

---

*"Building for the future where LLMs have million-token contexts and can reason across vast amounts of information. RAGAS is measuring for a world where we're scared of giving models too much context. That world is already obsolete."*