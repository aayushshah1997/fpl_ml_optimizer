#!/bin/bash
# Automated FPL data fetcher for future gameweeks
# Run this script weekly after each gameweek deadline to fetch new data

cd /Users/Aayush_1/Developer/FPL\ ML\ Model/fpl_ai

echo "🔄 Fetching latest FPL data..."

# Fetch data for the specified gameweek (or auto-detect)
if [ $# -eq 0 ]; then
    echo "Auto-detecting current gameweek..."
    python scripts/fetch_fpl_data.py --auto
else
    echo "Fetching data for GW$1..."
    python scripts/fetch_fpl_data.py $1
fi

if [ $? -eq 0 ]; then
    echo "✅ Data fetched successfully!"
    
    # Regenerate actual results
    echo "🔄 Regenerating actual results..."
    python scripts/generate_real_actual_results.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Actual results regenerated!"
        
        # Verify data quality
        echo "🔍 Verifying data quality..."
        python scripts/verify_actual_results.py
        
        echo "🎉 All done! Dashboard is ready with latest data."
    else
        echo "❌ Failed to regenerate actual results"
        exit 1
    fi
else
    echo "❌ Failed to fetch FPL data"
    exit 1
fi
