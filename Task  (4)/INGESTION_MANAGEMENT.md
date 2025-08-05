# Ingestion Management Scripts

This directory contains scripts to manage and monitor the Qdrant vector database ingestion process for the Freelancer Negotiation Helper.

## Quick Status Check

```bash
./check_ingestion_status.sh
```

This script provides a quick overview of:
- Current point counts in both collections (naive and advanced)
- Whether ingestion is currently running
- Container name if running

## Full Ingestion Manager

```bash
python3 check_and_ingest.py
```

This interactive Python script:
1. **Checks Current Status**: Shows points in both collections
2. **Determines if Ingestion Needed**: Alerts if advanced collection is empty
3. **Manages Ingestion Process**: 
   - Stops any existing ingestion
   - Starts fresh ingestion
   - Monitors progress in real-time
4. **Tracks Progress**:
   - Shows embedding generation count
   - Displays success/error messages
   - Waits for completion (up to 10 minutes)
5. **Provides Final Status**: Shows final point counts

### Features:
- ✅ Color-coded output for easy reading
- 📊 Real-time progress tracking
- 🔄 Automatic detection of running processes
- ⏱️ Configurable timeouts and check intervals
- 🛡️ Error handling and recovery

## Manual Commands

If you prefer manual control:

### Start Ingestion
```bash
docker-compose up ingestion
```

### Monitor Logs
```bash
# Find container name
docker ps --filter "name=ingestion" --format "{{.Names}}"

# Follow logs
docker logs -f <container_name>
```

### Check Collections via API
```bash
# Check naive collection
curl -s "http://localhost:6333/collections/never_split_naive" | jq '.result.points_count'

# Check advanced collection  
curl -s "http://localhost:6333/collections/never_split_advanced" | jq '.result.points_count'
```

## Expected Results

After successful ingestion:
- **Naive Collection**: ~205 points (fixed-size chunks)
- **Advanced Collection**: 50-100+ points (hierarchical parent-child chunks)

## Troubleshooting

### Ingestion Takes Too Long
- Normal duration: 5-10 minutes
- Depends on OpenAI API rate limits
- Check logs for errors: `docker logs $(docker ps -qf name=ingestion)`

### Advanced Collection Remains Empty
1. Check for errors in logs
2. Verify OpenAI API key is valid
3. Ensure `/data/output.md` exists
4. Try restarting: `docker-compose restart ingestion`

### Connection Errors
- Ensure Qdrant is running: `docker ps | grep qdrant`
- Check port 6333 is accessible
- Verify Docker network: `docker network ls`

## How It Works

1. **Dual Chunking Strategy**:
   - Naive: Fixed 500-token chunks with 50-token overlap
   - Advanced: Hierarchical parent-child chunks based on markdown structure

2. **Embedding Generation**:
   - Uses OpenAI's text-embedding-3-small model
   - Generates embeddings for each chunk
   - Stores in respective Qdrant collections

3. **Progress Monitoring**:
   - Tracks API calls to estimate progress
   - Monitors container logs for success/error messages
   - Checks collection point counts periodically