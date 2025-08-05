### 1. Proposed solution (what the freelancer sees)

The app will be a single-page chat window styled with **shadcn/ui** components.
At the top, the user pastes their OpenAI API key once; beneath that they drop or paste any draft email or chat transcript. A toggle labelled **“Naïve RAG ↔ Advanced RAG”** reruns the prompt so the freelancer can compare answers side-by-side. Both modes draw from the same knowledge base (the full text of *Never Split the Difference*), letting the user feel the difference in real time. Responses arrive as color-coded advice (“mirror,” “label emotion,” “calibrated question,” etc.) with copy-ready rewrites they can paste straight back to the client. The whole experience is as fast and friction-free as any consumer chat app, yet powered by a containerized, agent-driven backend.

### 2. Technology choices and why — **updated**

| Layer                             | Tool / Model                        | One-sentence rationale                                                                                                                                              |
| --------------------------------- | ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LLM – Supervisor Agent**        | **`gpt-4.1` (OpenAI)**              | Flagship 2025 model with 1 M-token context and top-tier reasoning, ideal for routing tasks and synthesising multi-agent outputs. ([OpenAI][1])                      |
| **LLM – Web Search & RAG Agents** | **`gpt-4.1-mini` (OpenAI)**         | 83 % cheaper and \~2× faster than GPT-4o while matching its benchmark scores—perfect for high-frequency scraping and summarisation jobs. ([OpenAI][1], [OpenAI][2]) |
| **LLM – Synthesis Agent**         | **`gpt-4.5` (OpenAI)**              | Latest large-capacity model with upgraded “EQ” and writing polish, making it the best choice for Chris-Voss-style, client-ready prose. ([OpenAI][3])                |
| Embeddings                        | **`text-embedding-3-small`**        | 5× cheaper than `ada-002` yet state-of-the-art on MTEB—keeps vector storage costs negligible without hurting recall. ([OpenAI Platform][4])                         |
| Orchestration                     | **LangGraph**                       | Graph-native workflows give explicit state and concurrency control for multi-agent patterns. ([LangChain AI][5])                                                    |
| Vector DB                         | **Qdrant (in Docker)**              | Rust-based vector engine with an official Docker image and built-in Web UI—simple local spin-up and blazing search. ([Qdrant][6])                                   |
| Monitoring                        | **LangSmith**                       | First-class tracing, dashboards, cost tracking and alerts for LangGraph/LangChain apps. ([docs.smith.langchain.com][7])                                             |
| Evaluation                        | **Ragas (in Docker micro-service)** | Purpose-built RAG/agent metrics with a LangGraph integration endpoint for automated CI scoring. ([docs.ragas.io][8])                                                |
| UI                                | **shadcn/ui + Tailwind**            | Pre-styled, accessible React components that match Tailwind tokens—cuts front-end build time. ([LangChain AI][5])                                                   |


[1]: https://openai.com/index/gpt-4-1/ "Introducing GPT-4.1 in the API | OpenAI"
[2]: https://openai.com/index/gpt-4-1/?utm_source=chatgpt.com "Introducing GPT-4.1 in the API - OpenAI"
[3]: https://openai.com/index/introducing-gpt-4-5/ "Introducing GPT-4.5 | OpenAI"
[4]: https://platform.openai.com/docs/pricing?utm_source=chatgpt.com "Pricing - OpenAI API"
[5]: https://langchain-ai.github.io/langgraph/?utm_source=chatgpt.com "LangGraph - GitHub Pages"
[6]: https://qdrant.tech/documentation/quickstart/?utm_source=chatgpt.com "Local Quickstart - Qdrant"
[7]: https://docs.smith.langchain.com/?utm_source=chatgpt.com "Get started with LangSmith | 🦜️🛠️ LangSmith"
[8]: https://docs.ragas.io/en/v0.2.7/howtos/integrations/_langgraph_agent_evaluation/?utm_source=chatgpt.com "LangGraph - Ragas"


### 3. Where agents fit 

| Role                                     | Core Responsibilities                                                                                                                                                                                                                                                                                                   | Typical Tools / Calls                                                                                  |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Supervisor Agent** (dispatcher)        | • Interprets the user’s request, asks follow-up questions if context is missing.<br>• Launches and tracks parallel subtasks for Web Search and RAG Search agents.<br>• Hands their structured outputs to the Synthesis Agent.                                                                                           | LangGraph state node with memory; ReAct tool picker (`web_search`, `rag_search`).                      |
| **Web Search Agent**                     | • Scrapes live sources for current market and wage data tailored to the freelancer’s skillset, region, and project type.<br>• Returns a concise `MarketRateReport` (ranges, median, links).                                                                                                                             | SerpAPI (or similar) → JSON schema.                                                                    |
| **RAG Search Agent**                     | • Converts the user’s question into refined queries.<br>• Retrieves passages from the Qdrant vector store of *Never Split the Difference* and other negotiation texts.<br>• Summarizes key tactics and objections into `NegotiationInsights`.                                                                           | LangChain Retriever → Qdrant; summarization via `gpt-4.1-mini`.                                        |
| **Synthesis Agent** (Chris Voss stylist) | • Consumes `MarketRateReport` + `NegotiationInsights` + original user draft.<br>• Crafts the final deliverable in Chris Voss’s voice—mirrors, labels, calibrated questions, “that’s right” moments.<br>• Outputs: (a) strategy checklist, (b) email/talk-track rewrite ready to paste, (c) 2-line market-rate snapshot. | Prompt-templated call to `gpt-4.1-mini` with elevated temperature; style guardrails via system prompt. |

**Agentic flow**

1. **User query** → Supervisor.
2. Supervisor clarifies details if needed, then fires two asynchronous tasks.
3. **Web Search Agent** returns `MarketRateReport`; **RAG Search Agent** returns `NegotiationInsights`.
4. Supervisor forwards both payloads (plus the user’s draft) to the **Synthesis Agent**.
5. **Synthesis Agent** produces a Chris-Voss-styled response combining strategy, rewritten text, and market data.
6. Supervisor sends that polished answer back to the user.

Adding the dedicated Synthesis Agent cleanly separates rhetoric from data gathering: the sub-agents focus on facts, while the stylist turns those facts into persuasive language that sounds like a seasoned hostage negotiator guiding the freelancer to “leave nothing on the table.”


[1]: https://openai.com/index/gpt-4-1/?utm_source=chatgpt.com "Introducing GPT-4.1 in the API - OpenAI"
[2]: https://openai.com/index/new-embedding-models-and-api-updates/?utm_source=chatgpt.com "New embedding models and API updates - OpenAI"
[3]: https://langchain-ai.github.io/langgraph/concepts/multi_agent/?utm_source=chatgpt.com "LangGraph Multi-Agent Systems - Overview"
[4]: https://qdrant.tech/documentation/quickstart/?utm_source=chatgpt.com "Local Quickstart - Qdrant"
[5]: https://docs.smith.langchain.com/?utm_source=chatgpt.com "Get started with LangSmith | 🦜️🛠️ LangSmith - LangChain"
[6]: https://docs.ragas.io/en/stable/howtos/integrations/_langgraph_agent_evaluation/ "LangGraph - Ragas"
[7]: https://ui.shadcn.com/docs/installation/manual?utm_source=chatgpt.com "Manual Installation - Shadcn UI"
