# Quick Reference Card

## Most Common Commands

```bash
# 🚀 Run complete pipeline (traditional)
python load_to_render_postgres.py

# 📋 Run step by step (recommended for production)
python load_to_render_postgres.py --step fetch --num-records 10000
python load_to_render_postgres.py --step prepare
python load_to_render_postgres.py --step embed
python load_to_render_postgres.py --step load

# 🔄 Resume from last checkpoint (after interruption)
python load_to_render_postgres.py --resume

# 📂 Check what checkpoints exist
python load_to_render_postgres.py --list-checkpoints

# 🗑️ Clean up old checkpoints
python load_to_render_postgres.py --cleanup-checkpoints
```

## When to Use Each Step

| Step | When to Use | Creates Checkpoint |
|------|-------------|-------------------|
| `fetch` | Get fresh data from API | ✅ `fetch_*.parquet` |
| `prepare` | Clean/transform data | ✅ `prepare_*.parquet` |
| `embed` | Generate embeddings (💰 costs money!) | ✅ `embed_*.parquet` |
| `load` | Insert into database | ❌ |
| `all` | Run everything at once | ❌ (no checkpoints) |

## Recovery Scenarios

| Problem | Solution |
|---------|----------|
| Fetch failed halfway | Re-run `--step fetch` |
| Prepare failed | Fix issue, run `--step prepare` |
| Embed failed (expensive!) | Fix issue, run `--step embed` (uses prepare checkpoint) |
| Load failed (DB issue) | Fix DB, run `--step load` (embeddings safe!) |
| Process interrupted | Run `--resume` |

## Useful Options

```bash
# Fetch fewer records (testing)
--num-records 100

# Skip embedding generation (faster, cheaper)
--no-embeddings

# Create table if missing
--create-table

# Don't save checkpoints (not recommended)
--no-checkpoint

# Use different checkpoint directory
--checkpoint-dir /path/to/checkpoints
```

## Checkpoint Files Location

```
checkpoints/
├── metadata.json                    # Track all checkpoints
├── fetch_YYYYMMDD_HHMMSS.parquet   # Raw data from API
├── prepare_YYYYMMDD_HHMMSS.parquet # Cleaned data
└── embed_YYYYMMDD_HHMMSS.parquet   # Data with embeddings (💾 large!)
```

## Pro Tips

1. **Always use checkpoints for production** - saves time and money
2. **Embed step is expensive** - don't re-run unless necessary
3. **Load step can be retried** - it won't regenerate embeddings
4. **Use --resume** - it's smart and knows what to do
5. **Clean up old checkpoints** - they can get large

## Help

```bash
# Get full help
python load_to_render_postgres.py --help

# Read the guides
cat STEPIFIED_PIPELINE.md
cat IMPLEMENTATION_SUMMARY.md
```
