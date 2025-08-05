# Deployment Guide

## Quick Start

### 1. Install Dependencies
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### 2. Configure Environment
```bash
# Copy environment template
cp .env.template .env

# Edit .env and add your API keys:
# - OPENAI_API_KEY
# - COHERE_API_KEY (optional)
```

### 3. Run Evaluation

#### Recommended: Retrieval-Focused Evaluation
```bash
# Evaluate retrieval methods with appropriate metrics
uv run python main.py evaluate

# Quick test (3 questions, 3 retrievers)
uv run python main.py test
```

#### Generate Golden Dataset Only
```bash
uv run python main.py generate-golden --size 20
```

#### Use Custom PDF
```bash
uv run python main.py evaluate --pdf /path/to/document.pdf
```

## Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY . .

# Install dependencies
RUN uv sync --frozen

# Set environment variables
ENV PYTHONPATH=/app

# Run evaluation
CMD ["uv", "run", "python", "main.py", "evaluate"]
```

### Environment Variables

Required:
- `OPENAI_API_KEY` - OpenAI API key for GPT-4o

Optional:
- `OPENAI_MODEL` - Model to use (default: gpt-4o)
- `RETRIEVAL_K` - Number of docs to retrieve (default: 10)
- `CHUNK_SIZE` - Document chunk size (default: 750)

### Performance Optimization

The evaluation is optimized with:
- Parallel processing (4 workers)
- Reduced timeouts (60s)
- Limited retries (3)
- Efficient metric selection

Expected runtime:
- Full evaluation: ~45-60 minutes
- Quick test: ~10-15 minutes

### API Usage Estimates

Per full evaluation:
- API calls: ~500-800 (retrieval metrics only)
- Cost: ~$5-10 depending on document size
- Tokens: ~500k-1M

### Monitoring

Check progress in real-time:
- Progress bars show evaluation status
- Results saved to `outputs/` directory
- Logs show any errors or warnings

### Output Files

```
outputs/
├── retrieval_evaluation_results.csv    # Main results
├── diagrams/                           # Visualizations
│   ├── performance_heatmap.png
│   ├── cost_vs_performance.html
│   └── radar_chart.html
└── tables/                            # Comparison tables
    └── comprehensive_comparison.html
```

## Troubleshooting

### Slow Evaluation
- Reduce questions in golden dataset
- Use fewer retrievers for testing
- Check API rate limits

### API Errors
- Verify API key is correct
- Check rate limits
- Ensure sufficient credits

### Memory Issues
- Process smaller PDF files
- Reduce chunk size
- Use fewer parallel workers

## Best Practices

1. **Test First**: Always run quick test before full evaluation
2. **Monitor Costs**: Track API usage during evaluation
3. **Cache Results**: Save evaluation outputs for comparison
4. **Version Control**: Tag evaluations with git commits

## Support

For issues or questions:
1. Check `docs/` folder for detailed documentation
2. Review error messages in console output
3. Check API documentation for metric details
