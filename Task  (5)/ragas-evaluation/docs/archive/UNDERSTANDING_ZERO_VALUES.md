# Understanding Zero Values in RAGAS Metrics

## Overview
Zero values in RAGAS metrics are **not bugs** but legitimate scores that indicate specific conditions. Each metric has valid reasons for returning 0.0.

## When Metrics Return 0.0

### 1. Answer Relevancy (0.0 when):
- **Empty response**: The model returns an empty string
- **Very short response**: Single word or extremely brief answers (e.g., "Yes", "No", "AI")
- **Completely irrelevant response**: Answer has no semantic relation to the question

**Example:**
```
Question: "What is machine learning?"
Response: "" or "ML" or "The weather is nice"
Result: 0.0
```

### 2. Context Precision (0.0 when):
- **No retrieved contexts**: The retriever returns an empty list
- **Completely irrelevant contexts**: Retrieved contexts have no relation to the question
- **No overlap**: Retrieved contexts don't support the response at all

**Example:**
```
Question: "What is machine learning?"
Retrieved Contexts: [] or ["Paris is the capital of France"]
Result: 0.0
```

### 3. Answer Correctness (0.0 when):
- **Complete mismatch**: Response has no factual overlap with reference
- **Wrong information**: Response contradicts the reference entirely
- **No semantic similarity**: Response and reference are about different topics

### 4. LLM Context Recall (0.0 when):
- **No claim matches**: Claims in reference don't appear in retrieved contexts
- **Different phrasing**: Even slight variations prevent matching
- **Context mismatch**: Retrieved contexts are from different parts of the document

### 5. Factual Correctness (0.0 when):
- **No factual overlap**: Response facts don't match reference facts
- **Different claim structure**: Claims are structured differently
- **Paraphrasing**: Even correct paraphrases may not match

## Why This Is Expected Behavior

1. **Precision Metrics**: These metrics measure exact matches and precision, not general quality
2. **Strict Evaluation**: They use strict criteria to ensure high-quality retrieval
3. **Real-World Scenarios**: In production, retrievers may fail to find relevant contexts
4. **Quality Indicators**: 0.0 values help identify when retrieval or generation fails

## What This Means for Evaluation

When you see 0.0 values:
1. **Check the retriever**: Is it finding relevant contexts?
2. **Check the response**: Is it answering the question?
3. **Check the reference**: Does it match what was retrieved?
4. **Consider the metric**: Is this metric appropriate for your use case?

## Recommendations

1. **Don't Hide 0.0 Values**: They provide important diagnostic information
2. **Use Multiple Metrics**: No single metric tells the whole story
3. **Analyze Patterns**: If a retriever consistently scores 0.0, investigate why
4. **Context Matters**: Some metrics work better for certain types of questions

## Example Analysis

```python
# Good performance despite some 0.0s:
{
    "faithfulness": 0.92,          # Response faithful to contexts
    "semantic_similarity": 0.84,    # Good semantic match
    "answer_relevancy": 0.85,       # Relevant answer
    "context_precision": 0.75,      # Good context selection
    "llm_context_recall": 0.0,      # Reference not in contexts (expected)
    "factual_correctness": 0.0      # Different phrasing (expected)
}
```

This shows a well-performing system where the 0.0 values are expected due to the strict nature of those specific metrics.