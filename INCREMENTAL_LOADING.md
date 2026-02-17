# Incremental Loading Feature

## Overview

The pipeline has been updated to support **incremental loading**, which avoids re-processing records that already exist in the database. This saves significant time and API costs by skipping:

1. ✅ Records that already exist in the database
2. 🧠 Embedding generation for existing records
3. 💾 Database insertion attempts for duplicates

## What Changed

### 1. New Method: `get_existing_nct_ids()`

This method queries the database to get all NCT IDs that already exist:

```python
def get_existing_nct_ids(self, conn: psycopg.Connection) -> set:
    """Get set of NCT IDs that already exist in the database"""
```

- Checks if the `clinical_trials` table exists
- Returns a set of all existing NCT IDs
- Returns empty set if table doesn't exist

### 2. Enhanced `prepare_data()` Method

Now accepts an optional `existing_nct_ids` parameter:

```python
def prepare_data(self, df: pd.DataFrame, existing_nct_ids: Optional[set] = None) -> Optional[pd.DataFrame]:
```

- Filters out records with NCT IDs that already exist
- Logs how many records were skipped
- Returns `None` if all records already exist

### 3. Updated `run_full_pipeline()` Method

Added new parameter `skip_existing`:

```python
def run_full_pipeline(
    self,
    num_records: int = 10,
    generate_embeddings: bool = False,
    create_table: bool = False,
    drop_table_first: bool = False,
    filter_by_start_date: bool = False,
    start_year: int = 2025,
    skip_existing: bool = True  # NEW!
) -> Dict[str, Any]:
```

- When `skip_existing=True`, it queries existing NCT IDs before processing
- Only processes new records
- Returns summary with `existing_skipped` count

### 4. Updated Configuration

Changed default settings for incremental loading:

```python
# Database operations
DROP_EXISTING_TABLE = False  # Changed from True - preserve existing data
CREATE_TABLE = True
LOAD_TO_DATABASE = True

# Skip existing records (NEW!)
SKIP_EXISTING_RECORDS = True
```

## How It Works

### Previous Behavior (Without Skip)
```
API Fetch (75,189 records)
    ↓
Prepare Data (75,189 records)
    ↓
Generate Embeddings (75,189 × $0.0001 = $7.52) 💸
    ↓
Load to Database
    ↓
ON CONFLICT DO NOTHING (duplicates wasted)
```

### New Behavior (With Skip)
```
API Fetch (75,189 records)
    ↓
Check Database (63,120 already exist)
    ↓
Prepare Data (12,069 NEW records only)
    ↓
Generate Embeddings (12,069 × $0.0001 = $1.21) 💰
    ↓
Load to Database (only new records)
```

## Cost Savings Example

Based on your log file:
- **Total fetched:** 75,189 records
- **Already in DB:** 63,120 records (84%)
- **New records:** 12,069 records (16%)

### Without Skip Feature:
- API calls for embeddings: 75,189 texts
- Estimated cost: ~$7.52

### With Skip Feature:
- API calls for embeddings: 12,069 texts
- Estimated cost: ~$1.21
- **Savings: ~$6.31 (84%)**

Plus time saved:
- Without: ~30 minutes for embedding generation
- With: ~5 minutes for embedding generation
- **Time saved: ~25 minutes (83%)**

## Usage

### To Use Incremental Loading (Recommended):
```python
# In configuration section
DROP_EXISTING_TABLE = False  # Keep existing data
SKIP_EXISTING_RECORDS = True  # Skip processing existing records
```

### To Re-process Everything:
```python
# In configuration section
DROP_EXISTING_TABLE = True   # Start fresh
SKIP_EXISTING_RECORDS = False # Process all records
```

Or:

```python
DROP_EXISTING_TABLE = False  # Keep table
SKIP_EXISTING_RECORDS = False # But re-process everything (uses ON CONFLICT DO NOTHING)
```

## Output Example

With the new feature, you'll see:

```
🔍 Checking for existing records...
✅ Found 63120 existing records in database

📥 STEP 1: Fetching data...
✅ Fetched 75189 records

🔄 STEP 2: Preparing data...
⏭️  Skipping 63120 existing records, processing 12069 new records
✅ Prepared 12069 records

🧠 STEP 3: Generating embeddings...
✅ Generated 12069/12069 embeddings

================================================================================
EXECUTION SUMMARY
================================================================================
✅ Fetched:      75189 records
⏭️  Skipped:      63120 existing records
✅ Prepared:     12069 records
✅ Embeddings:   Generated
✅ Loaded:       12069 records
================================================================================
```

## Database Behavior

The code uses PostgreSQL's `ON CONFLICT DO NOTHING`:

```sql
INSERT INTO clinical_trials (...)
VALUES (...)
ON CONFLICT (nct_id)
DO NOTHING
```

This means:
- If you set `SKIP_EXISTING_RECORDS = False`, duplicates are safely ignored
- No errors occur, duplicates just don't get inserted
- Existing records are NOT updated

## Answering Your Question

> "If I re-run the process will it re-do everything or can I only insert the rows I didn't already insert?"

**Answer:** With the new `SKIP_EXISTING_RECORDS = True` setting (now the default):

1. ✅ **It will ONLY process new records**
2. ✅ **It will NOT generate embeddings for existing records**
3. ✅ **It will NOT waste API calls**
4. ✅ **It will save you time and money**

The pipeline:
- Checks the database for existing NCT IDs
- Filters them out during the prepare step
- Only generates embeddings for new records
- Only inserts new records

So you can safely re-run the script with the same parameters, and it will pick up where it left off!
