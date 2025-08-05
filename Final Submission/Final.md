# AIM Certificate Challenge - Project Index

## Project Overview

This repository contains my complete submission for the AIM Certificate Challenge, showcasing the development of an advanced RAG (Retrieval-Augmented Generation) system focused on negotiation assistance based on Chris Voss's "Never Split the Difference."

### Key Innovation: Parent-Child Retrieval Architecture

While traditional RAG systems optimize for precision metrics, this project explores a forward-thinking approach that embraces large context windows and hierarchical document understanding. The Parent-Child chunking strategy, though scoring lower on traditional RAGAS metrics, represents a bet on the future of LLM capabilities.

## Project Navigation

### Core Tasks

1. **[Task 1: Initial Exploration](../Task%20%20(1)/Task_1.md)**
   - Initial project setup and understanding
   - Baseline requirements analysis

2. **[Task 2: System Design](../Task%20%20(2)/Task_2.md)**
   - Architecture planning
   - Technology stack decisions

3. **[Task 3: Implementation Planning](../Task%20%20(3)/Task_03.md)**
   - Detailed implementation roadmap
   - Component specifications

4. **Task 4: Advanced RAG Implementation**
   - **[Main Project README](../Task%20%20(4)/README.md)** - Complete system overview
   - **[Build Plan](../Task%20%20(4)/build%20plan.md)** - Detailed implementation strategy
   - **Key Documentation:**
     - [Advanced Child-Parent Retriever Documentation](../Task%20%20(4)/ADVANCED_CHILD_PARENT_RETRIEVER_CHUNKING.md)
     - [Parent-Child Implementation Guide](../Task%20%20(4)/PARENT_CHILD_CHUNKING_IMPLEMENTATION_GUIDE.md)
     - [Ingestion System Documentation](../Task%20%20(4)/backend/ingestion/INGESTION_README.md)
     - [Ingestion Pipeline Updates](../Task%20%20(4)/backend/ingestion/INGESTION_PIPELINE_UPDATES.md)

5. **[Task 5: Performance Analysis](../Task%20%20(5)/Task_05.md)**
   - Deep dive into why Parent-Child scored poorly on RAGAS
   - Analysis of the embedding-retrieval mismatch
   - Insights on evaluation metrics vs real-world performance

6. **[Task 6: Retrieval Techniques Comparison](../Task%20%20(6)/Task_06.md)**
   - Tested 8 different retrieval strategies
   - Ensemble method emerged as the winner
   - Parent-Child approach: ambitious but flawed execution

7. **[Task 7: Future Directions](../Task%20%20(7)/Task_07.md)**
   - Performance assessment of naive vs advanced approaches
   - Vision for embracing large context windows
   - Challenging traditional evaluation paradigms

## Key Technical Components

### Backend Services
- **[API Service](../Task%20%20(4)/backend/api/README.md)** - FastAPI-based REST endpoints
- **[Ingestion Service](../Task%20%20(4)/backend/ingestion/README.md)** - Dual chunking pipeline
- **[Evaluation Service](../Task%20%20(4)/backend/evaluation/README.md)** - RAGAS-based evaluation

### Core Innovations

1. **Dual Chunking Strategy**
   - Naive chunking for baseline comparison
   - Advanced Parent-Child hierarchical chunking
   - Parallel processing for performance analysis

2. **Context Corridor Enhancement**
   - Recognizes when document sections flow into each other
   - Retrieves additional context when similarity threshold is met
   - Designed for narrative continuity

3. **Large Context Philosophy**
   - Built for 128k+ token context windows
   - Challenges precision-focused metrics
   - Prioritizes comprehensive understanding

## Project Philosophy

> "Sometimes the best metric is asking a simple question: 'Which answer would I rather receive?'"

This project represents a philosophical stance: that the future of RAG systems lies not in retrieving the perfect minimal chunk, but in providing rich context that modern LLMs can navigate effectively. While RAGAS metrics penalize this approach, real-world usage suggests users prefer comprehensive, nuanced answers.

## Quick Start

```bash
# Clone the repository
git clone [repository-url]

# Navigate to Task 4 (main implementation)
cd "Task  (4)"

# Set up environment
cp .env.example .env
# Add your API keys to .env

# Run with Docker
docker-compose up

# Or run locally
cd backend/api
python main.py
```

## Key Findings

1. **The Embedding-Retrieval Mismatch**: Searching on small chunks while returning large chunks fundamentally breaks precision metrics
2. **RAGAS Limitations**: Current metrics are optimized for small context windows, not modern LLM capabilities
3. **Ensemble Superiority**: Simple combination of BM25 + dense retrieval outperformed complex approaches
4. **Context vs Precision Trade-off**: More context often means better answers but worse metrics

## Lessons Learned

- Sometimes simple > complex (Ensemble beat everything)
- Metrics aren't everything (user preference != RAGAS scores)
- Build for tomorrow's LLMs, not yesterday's constraints
- Test with your actual evaluation criteria early

## Future Vision

The Parent-Child approach, while "failing" by traditional metrics, points toward a future where:
- LLMs have million-token contexts
- Retrieval provides rich, interconnected knowledge graphs
- Systems optimize for understanding, not just precision
- Evaluation metrics evolve to match capabilities

---

*"Building for the future where LLMs have million-token contexts and can reason across vast amounts of information. RAGAS is measuring for a world where we're scared of giving models too much context. That world is already obsolete."*