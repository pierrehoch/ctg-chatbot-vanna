# Database Configuration Requirements by Step

## Steps that DON'T need database configuration:

✅ **fetch** - Only fetches from ClinicalTrials.gov API
✅ **embed** - Only generates embeddings using OpenAI API  
✅ **prepare** (with `--no-skip-existing`) - Only data transformation

These steps can run without setting any database environment variables!

## Steps that NEED database configuration:

❗ **prepare** (with `--skip-existing`, which is the default) - Queries DB to check existing records
❗ **load** - Inserts data into database
❗ **all** - Runs all steps including load
❗ **--resume** - May need to check database depending on which step is next

## How to Set Database Configuration

### Option 1: Database URL (Recommended)
```bash
export DATABASE_URL="postgresql://user:password@host:port/database"
# OR
export RENDER_DATABASE_URL="postgresql://user:password@host:port/database"
```

### Option 2: Individual Credentials
```bash
export RENDER_DB_HOST="your-host.render.com"
export RENDER_DB_NAME="your_database_name"
export RENDER_DB_USER="your_username"
export RENDER_DB_PASSWORD="your_password"
export RENDER_DB_PORT="5432"  # Optional, defaults to 5432
```

## Example Workflows

### Workflow 1: Fetch and embed without database
```bash
# No database config needed!
python load_to_render_postgres.py --step fetch --num-records 10000
python load_to_render_postgres.py --step prepare --no-skip-existing
python load_to_render_postgres.py --step embed

# Now set database config and load
export DATABASE_URL="postgresql://..."
python load_to_render_postgres.py --step load
```

### Workflow 2: Complete pipeline with database from start
```bash
# Set database config once
export DATABASE_URL="postgresql://..."

# Run all steps
python load_to_render_postgres.py --step fetch --num-records 10000
python load_to_render_postgres.py --step prepare  # Will skip existing records
python load_to_render_postgres.py --step embed
python load_to_render_postgres.py --step load
```

### Workflow 3: Testing without database
```bash
# Great for development/testing!
python load_to_render_postgres.py --step fetch --num-records 10
python load_to_render_postgres.py --step prepare --no-skip-existing
python load_to_render_postgres.py --step embed --no-embeddings
# Stop here, test your checkpoints, verify data format, etc.
```

## Pro Tip

If you're doing development work and don't have database access yet, you can still:
1. Fetch data
2. Prepare/clean data  
3. Generate embeddings
4. Save everything to checkpoints

Then later, when you have database access, just run the load step!
