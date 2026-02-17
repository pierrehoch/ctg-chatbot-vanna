# Stepified Pipeline Guide

The clinical trials data pipeline now supports **step-by-step execution** with automatic checkpoint management. This allows you to run individual steps and resume from failures without losing progress.

## 🎯 Key Features

- ✅ **Individual step execution** - Run fetch, prepare, embed, or load separately
- ✅ **Automatic checkpoints** - Data is saved after each step
- ✅ **Resume capability** - Automatically continue from last checkpoint
- ✅ **Progress preservation** - Never lose expensive embedding work
- ✅ **Backward compatible** - Old usage still works with `--step all`

## 📋 Pipeline Steps

1. **fetch** - Fetch data from ClinicalTrials.gov API
2. **prepare** - Clean, transform, and prepare data
3. **embed** - Generate OpenAI embeddings (expensive!)
4. **load** - Insert data into PostgreSQL database

## 🚀 Quick Start

### Run Complete Pipeline (Traditional Way)
```bash
python load_to_render_postgres.py --step all
```

### Run Individual Steps
```bash
# Step 1: Fetch data from API
python load_to_render_postgres.py --step fetch --num-records 10000

# Step 2: Prepare/clean data
python load_to_render_postgres.py --step prepare

# Step 3: Generate embeddings (can take a while!)
python load_to_render_postgres.py --step embed

# Step 4: Load to database
python load_to_render_postgres.py --step load --create-table
```

### Resume from Last Checkpoint
```bash
# Automatically detects last completed step and continues
python load_to_render_postgres.py --resume
```

## 💾 Checkpoint Management

### List Available Checkpoints
```bash
python load_to_render_postgres.py --list-checkpoints
```

### Cleanup Old Checkpoints
```bash
# Keeps last 3 checkpoints for each step, removes older ones
python load_to_render_postgres.py --cleanup-checkpoints
```

## 📁 Checkpoint Storage

Checkpoints are stored in the `checkpoints/` directory:
- `fetch_YYYYMMDD_HHMMSS.parquet` - Raw fetched data
- `prepare_YYYYMMDD_HHMMSS.parquet` - Cleaned/prepared data
- `embed_YYYYMMDD_HHMMSS.parquet` - Data with embeddings
- `metadata.json` - Checkpoint metadata

## 🔧 Common Use Cases

### 1. Large Data Fetch That Might Fail
```bash
# Fetch data (creates checkpoint)
python load_to_render_postgres.py --step fetch --num-records 500000

# If it fails partway, you still have the partial data
# Continue with prepare step
python load_to_render_postgres.py --step prepare
```

### 2. Expensive Embedding Generation
```bash
# Run fetch and prepare first
python load_to_render_postgres.py --step fetch --num-records 10000
python load_to_render_postgres.py --step prepare

# Generate embeddings (this is expensive!)
python load_to_render_postgres.py --step embed

# If embedding fails halfway, resume it:
python load_to_render_postgres.py --step embed
# (Loads from prepare checkpoint and regenerates embeddings)

# Once embeddings are done, load to database
python load_to_render_postgres.py --step load
```

### 3. Database Load Issues
```bash
# If load step fails due to database issues
# You can fix the database and retry without re-generating embeddings
python load_to_render_postgres.py --step load --create-table
```

### 4. Resume After Interruption
```bash
# If you interrupt the process (Ctrl+C) at any step:
python load_to_render_postgres.py --resume
# Automatically continues from where it left off
```

## ⚙️ Configuration Options

### Fetch Step Options
```bash
--num-records N              # Number of records to fetch (default: 500000)
--filter-by-start-date       # Filter by start date (default: enabled)
--no-filter-by-start-date    # Disable start date filter
--start-year YYYY            # Studies after this year (default: 2014)
```

### Prepare Step Options
```bash
--skip-existing              # Skip records already in DB (default: enabled)
--no-skip-existing           # Process all records
```

### Embed Step Options
```bash
--generate-embeddings        # Generate embeddings (default: enabled)
--no-embeddings              # Skip embeddings
```

### Load Step Options
```bash
--create-table               # Create table if missing
--drop-table                 # Drop existing table first
```

### Checkpoint Options
```bash
--checkpoint-dir DIR         # Checkpoint directory (default: checkpoints)
--no-checkpoint              # Don't save checkpoints
```

## 🎬 Example Workflows

### Full Pipeline with Checkpoints (Safest)
```bash
python load_to_render_postgres.py --step fetch --num-records 100000
python load_to_render_postgres.py --step prepare
python load_to_render_postgres.py --step embed
python load_to_render_postgres.py --step load
```

### Quick Run Without Checkpoints
```bash
python load_to_render_postgres.py --step all --no-checkpoint
```

### Resume After Failure
```bash
# Pipeline failed during embed step
# Just run resume - it will continue from prepare checkpoint
python load_to_render_postgres.py --resume
```

## 🔍 Monitoring Progress

Each step shows detailed progress:
- Fetch: Shows pages fetched and total records
- Prepare: Shows records processed and skipped
- Embed: Shows embeddings generated with progress
- Load: Shows batch insertion progress

Checkpoints are automatically saved after each successful step, so you can always resume!

## ⚠️ Important Notes

1. **Embeddings are expensive**: The embed step costs money (OpenAI API). Use checkpoints to avoid re-running!
2. **Checkpoint size**: Checkpoints can be large (especially with embeddings). Monitor disk space.
3. **Database connection**: Load step needs database credentials in environment variables.
4. **Resume logic**: Resume automatically detects the last completed step and continues from there.

## 🆘 Troubleshooting

### "No checkpoint found"
Run the previous step first:
```bash
# If prepare fails with "No checkpoint found"
python load_to_render_postgres.py --step fetch
python load_to_render_postgres.py --step prepare
```

### Database connection errors during load
Fix your database credentials and retry:
```bash
# Load step can be retried without losing embeddings
python load_to_render_postgres.py --step load
```

### Out of disk space
Clean up old checkpoints:
```bash
python load_to_render_postgres.py --cleanup-checkpoints
```

## 🎉 Benefits

- **No more losing progress** - Checkpoints preserve your work
- **Cost savings** - Don't regenerate expensive embeddings
- **Debugging friendly** - Test each step independently
- **Flexible execution** - Run steps when convenient
- **Production ready** - Resume after any failure
