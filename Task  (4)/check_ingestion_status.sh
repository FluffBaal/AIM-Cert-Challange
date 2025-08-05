#!/bin/bash
# Quick script to check Qdrant collection status

echo "=== Qdrant Collection Status ==="
echo ""

# Function to check collection
check_collection() {
    local collection=$1
    local response=$(curl -s "http://localhost:6333/collections/$collection" 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        local points=$(echo "$response" | grep -o '"points_count":[0-9]*' | cut -d':' -f2)
        local vectors=$(echo "$response" | grep -o '"vectors_count":[0-9]*' | cut -d':' -f2)
        
        if [ -n "$points" ]; then
            echo "$collection:"
            echo "  Points: $points"
            if [ "$points" -gt 0 ]; then
                echo "  Status: ✅ Populated"
            else
                echo "  Status: ⚠️  Empty"
            fi
        else
            echo "$collection: ❌ Not found"
        fi
    else
        echo "$collection: ❌ Cannot connect to Qdrant"
    fi
    echo ""
}

# Check both collections
check_collection "never_split_naive"
check_collection "never_split_advanced"

# Check if ingestion is running
echo "Ingestion Status:"
if docker ps | grep -q ingestion; then
    echo "  🔄 Currently running"
    container=$(docker ps --filter "name=ingestion" --format "{{.Names}}" | head -1)
    echo "  Container: $container"
    echo "  To monitor: docker logs -f $container"
else
    echo "  ⏸️  Not running"
fi

echo ""
echo "To start ingestion: docker-compose up ingestion"
echo "To check progress: python3 check_and_ingest.py"