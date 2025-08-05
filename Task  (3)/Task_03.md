### 1. Data sources & external API

| Source / API                                                        | What it contains                                                                                                                                                 | How we’ll use it                                                                                                                                                                                                                | Which agent uses it |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| **“Never Split the Difference” — Chris Voss (author-approved PDF)** | Full text of the 2016 negotiation classic: tactics (mirroring, labeling, calibrated questions), real-world dialogue examples, and objection-handling frameworks. | Ingest once, split into parent/child chunks, embed with `text-embedding-3-small`, and store in Qdrant. At run-time the **RAG Search Agent** retrieves the most relevant passages to build its `NegotiationInsights` brief.      | RAG Search Agent    |
| **Exa AI Search API** (`/search`, `/contents`, optional `/answer`)  | Real-time, LLM-oriented web search: returns ranked URLs, cleaned HTML snippets, and structured metadata. ([Exa][1])                                              | On demand the **Web Search Agent** queries Exa for “current freelance UX designer day-rate Germany 2025”, “typical revision rounds in SaaS contracts”, etc., then compiles a `MarketRateReport` (ranges, sources, date stamps). | Web Search Agent    |

This pairing gives us (1) deep, evergreen negotiation expertise and (2) fresh market numbers—exactly the two evidence streams the synthesis agent needs to craft persuasive, up-to-date advice.

---

### 2. Default chunking strategy — “Semantic-Throttled Children → Markdown-Aware Parents”

| Phase                                  | What happens (now that the source is **Markdown**)                                                                                                                                                                                                                          | Default numbers                                                                                                                      | Rationale                                                                                                        |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| **A. Heading pass (parent skeletons)** | Split on every top-level `#` or `##` heading.<br>Each section becomes an **initial parent** with a unique `parent_id`, `section_heading`, and `anchor_id` (a slug such as `#the-new-rules`).                                                                                | Hard ceiling **≤ 1 200 tokens** per parent; if a heading section exceeds this, sub-split at the nearest `###` or after \~800 tokens. | Markdown headings give us natural topic boundaries, so we no longer need a heavyweight community-detection step. |
| **B. Child semantic throttling**       | Inside each parent:<br>1 ➜ split on punctuation; 2 ➜ embed tentative slice; 3 ➜ **merge** with previous slice if cosine ≥ 0 .92; 4 ➜ **re-split** at next punctuation if cosine ≤ 0 .15.<br>**Never** merge across list starts (`*`, `1.`) or block-quote boundaries (`>`). | Target **100–140 tokens** per child chunk.                                                                                           | Preserves single-idea vectors while respecting Markdown structure (lists, quotes, dialogue).                     |
| **C. Thin-parent repair (optional)**   | If a heading section yields < 400 tokens after throttling, merge it with the next sibling section **only if** the cosine between their first children ≥ 0 .65.                                                                                                              | 400-token min                                                                                                                        | Prevents wafer-thin parents that add overhead without context.                                                   |
| **D. Store in Qdrant**                 | **Child collection** → vector + payload `{ parent_id, section_heading, anchor_id, child_idx }`.<br>**Parent storage** → full markdown text in the same payload (`parent_text`) or in a companion collection keyed by `parent_id`.                                           | Embedding: `text-embedding-3-small` (1536 dims).                                                                                     | Qdrant’s vector + payload model handles both layers without custom index logic.                                  |

---

#### Retrieval flow

1. **Similarity search** ↠ child vectors (precision first).
2. Qdrant **`group_by: "parent_id"`** (v ≥ 1 .11) returns the top-K parents with their best-matching child.
3. **Context corridor**: if the preceding parent’s last child has cosine ≥ 0 .55 with the query, prepend that parent to capture lead-in dialogue or set-ups.
4. Pass the selected parent markdown (with headings, bullets, quotes intact) to GPT-4 .1-mini.

---

#### Why this works

* **Cleaner parents, sharper children** – heading boundaries give self-contained mini-lessons; semantic throttling ensures each vector is a focused micro-topic.
* **Token-budget safe** – parents are capped at 1 200 tokens, so even two parents plus the user prompt fit comfortably in GPT-4 .1-mini’s context window.
* **Explainable citations** – `anchor_id` lets the chat response deep-link the exact source passage, boosting user trust.
* **Drop-in compatibility** – the `parent_id` map plugs straight into LangChain’s `ParentDocumentRetriever`; Qdrant’s native `group_by` keeps latency low and results uncluttered.

With Markdown-aware parents and semantic-throttled children, every retrieval surfaces a pinpoint idea from *Never Split the Difference* **and** the surrounding narrative the synthesis agent needs to craft persuasive, Chris-Voss-style guidance.


[1]: https://docs.exa.ai/reference/search?utm_source=chatgpt.com "Search - Exa"
[2]: https://python.langchain.com/docs/how_to/parent_document_retriever/?utm_source=chatgpt.com "How to use the Parent Document Retriever | 🦜️ LangChain"
[3]: https://medium.com/%40danushidk507/rag-ix-parent-document-retriever-a49450a482ab?utm_source=chatgpt.com "RAG IX — Parent Document Retriever | by DhanushKumar - Medium"
