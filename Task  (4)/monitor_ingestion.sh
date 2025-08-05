#!/bin/bash
# Real-time monitoring of ingestion progress

echo "Monitoring ingestion progress..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
    # Clear previous line
    echo -ne "\r\033[K"
    
    # Get collection counts
    naive=$(curl -s "http://localhost:6333/collections/never_split_naive" | grep -o '"points_count":[0-9]*' | cut -d':' -f2)
    advanced=$(curl -s "http://localhost:6333/collections/never_split_advanced" | grep -o '"points_count":[0-9]*' | cut -d':' -f2)
    
    # Check if container is running
    if docker ps | grep -q ingestion; then
        status="🔄 Running"
    else
        status="⏹️  Stopped"
    fi
    
    # Display status
    echo -ne "Status: $status | Naive: $naive | Advanced: $advanced"
    
    # If advanced is populated, we're done!
    if [ "$advanced" -gt "0" ]; then
        echo -e "\n\n✅ Ingestion complete! Advanced collection has $advanced points"
        break
    fi
    
    # If container stopped but no advanced points, something went wrong
    if [ "$status" = "⏹️  Stopped" ] && [ "$advanced" = "0" ]; then
        echo -e "\n\n❌ Ingestion stopped but advanced collection is still empty"
        echo "Check logs: docker logs task4-ingestion-1"
        break
    fi
    
    sleep 5
done