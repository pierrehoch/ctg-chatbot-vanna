#!/bin/bash
# Example usage of the stepified clinical trials pipeline

echo "=================================================="
echo "Clinical Trials Pipeline - Step by Step Examples"
echo "=================================================="

# Example 1: Run complete pipeline (traditional way)
echo -e "\n📌 Example 1: Complete pipeline"
echo "python load_to_render_postgres.py --step all --num-records 1000"

# Example 2: Run individual steps
echo -e "\n📌 Example 2: Step-by-step execution"
echo "# Step 1: Fetch data"
echo "python load_to_render_postgres.py --step fetch --num-records 10000"
echo ""
echo "# Step 2: Prepare data"
echo "python load_to_render_postgres.py --step prepare"
echo ""
echo "# Step 3: Generate embeddings"
echo "python load_to_render_postgres.py --step embed"
echo ""
echo "# Step 4: Load to database"
echo "python load_to_render_postgres.py --step load --create-table"

# Example 3: Resume from checkpoint
echo -e "\n📌 Example 3: Resume after interruption"
echo "python load_to_render_postgres.py --resume"

# Example 4: Large production run
echo -e "\n📌 Example 4: Large production run (500k records)"
echo "# Fetch all recent studies"
echo "python load_to_render_postgres.py --step fetch --num-records 500000 --start-year 2014"
echo ""
echo "# Prepare (this skips existing records in DB)"
echo "python load_to_render_postgres.py --step prepare"
echo ""
echo "# Generate embeddings (EXPENSIVE - costs money!)"
echo "python load_to_render_postgres.py --step embed"
echo ""
echo "# Load to database"
echo "python load_to_render_postgres.py --step load"

# Example 5: Testing/debugging workflow
echo -e "\n📌 Example 5: Testing with small dataset"
echo "# Fetch just 100 records for testing"
echo "python load_to_render_postgres.py --step fetch --num-records 100 --no-filter-by-start-date"
echo ""
echo "# Process without embeddings (faster for testing)"
echo "python load_to_render_postgres.py --step prepare"
echo "python load_to_render_postgres.py --step load --no-embeddings --create-table"

# Example 6: Checkpoint management
echo -e "\n📌 Example 6: Checkpoint management"
echo "# List all checkpoints"
echo "python load_to_render_postgres.py --list-checkpoints"
echo ""
echo "# Clean up old checkpoints"
echo "python load_to_render_postgres.py --cleanup-checkpoints"

# Example 7: Recovery scenarios
echo -e "\n📌 Example 7: Recovery from failures"
echo "# If embedding step fails halfway through:"
echo "# 1. Check the error"
echo "# 2. Fix the issue (API key, network, etc.)"
echo "# 3. Resume from prepare checkpoint:"
echo "python load_to_render_postgres.py --step embed"
echo ""
echo "# If load step fails due to DB issues:"
echo "# 1. Fix database connection"
echo "# 2. Retry load (embeddings are preserved!)"
echo "python load_to_render_postgres.py --step load"

echo -e "\n=================================================="
echo "For more details, see STEPIFIED_PIPELINE.md"
echo "=================================================="
