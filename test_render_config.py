#!/usr/bin/env python3
"""
Render PostgreSQL Configuration and Testing Utility

This utility helps you:
1. Verify your environment configuration
2. Test database connectivity
3. Create/drop tables
4. View data statistics
5. Debug connection issues

Usage:
    python test_render_config.py --help
    python test_render_config.py --check-env
    python test_render_config.py --test-connection
    python test_render_config.py --table-stats
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import psycopg
    PSYCOPG_AVAILABLE = True
except ImportError:
    PSYCOPG_AVAILABLE = False


class RenderConfigTester:
    """Utility for testing Render PostgreSQL configuration"""
    
    REQUIRED_ENV_VARS = [
        "RENDER_DB_HOST",
        "RENDER_DB_NAME",
        "RENDER_DB_USER",
        "RENDER_DB_PASSWORD"
    ]
    
    OPTIONAL_ENV_VARS = [
        "RENDER_DB_PORT",
        "OPENAI_API_KEY"
    ]
    
    def __init__(self):
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config = {}
        
        # Load required variables
        for var in self.REQUIRED_ENV_VARS:
            value = os.environ.get(var)
            config[var] = value
        
        # Load optional variables with defaults
        config["RENDER_DB_PORT"] = os.environ.get("RENDER_DB_PORT", "5432")
        config["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "Not set")
        
        return config
    
    def check_environment(self) -> bool:
        """Check if all required environment variables are set"""
        print("\n" + "="*80)
        print("ENVIRONMENT VARIABLE CHECK")
        print("="*80)
        
        all_set = True
        
        # Check required variables
        print("\n📋 Required Variables:")
        for var in self.REQUIRED_ENV_VARS:
            value = self.config.get(var)
            if value:
                masked_value = f"{value[:10]}..." if len(value) > 10 else value
                print(f"  ✅ {var}: {masked_value}")
            else:
                print(f"  ❌ {var}: NOT SET")
                all_set = False
        
        # Check optional variables
        print("\n📋 Optional Variables:")
        openai_key = self.config.get("OPENAI_API_KEY")
        if openai_key and openai_key != "Not set":
            print(f"  ✅ OPENAI_API_KEY: Set (masked)")
        else:
            print(f"  ⚠️  OPENAI_API_KEY: Not set (needed for embeddings)")
        
        # Summary
        print("\n" + "-"*80)
        if all_set:
            print("✅ All required variables are set!")
            return True
        else:
            print("❌ Some required variables are missing!")
            print("\nTo set them, run:")
            print("  export RENDER_DB_HOST='your-host'")
            print("  export RENDER_DB_NAME='your-db'")
            print("  export RENDER_DB_USER='your-user'")
            print("  export RENDER_DB_PASSWORD='your-password'")
            return False
    
    def test_connection(self) -> bool:
        """Test connection to the database"""
        print("\n" + "="*80)
        print("DATABASE CONNECTION TEST")
        print("="*80)
        
        if not PSYCOPG_AVAILABLE:
            print("❌ psycopg is not installed")
            print("   Install with: pip install psycopg[binary]")
            return False
        
        # Check environment first
        if not all(self.config.get(var) for var in self.REQUIRED_ENV_VARS):
            print("❌ Missing required environment variables")
            print("   Run: python test_render_config.py --check-env")
            return False
        
        try:
            print(f"\n🔌 Connecting to {self.config['RENDER_DB_HOST']}...")
            
            conn = psycopg.connect(
                host=self.config["RENDER_DB_HOST"],
                port=int(self.config["RENDER_DB_PORT"]),
                dbname=self.config["RENDER_DB_NAME"],
                user=self.config["RENDER_DB_USER"],
                password=self.config["RENDER_DB_PASSWORD"],
                sslmode="require"
            )
            
            print("✅ Connected successfully!")
            
            # Get server information
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                print(f"\n📊 PostgreSQL Version:")
                print(f"   {version}")
            
            # List tables
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """)
                tables = cur.fetchall()
                print(f"\n📋 Tables in database:")
                if tables:
                    for table in tables:
                        print(f"   - {table[0]}")
                else:
                    print("   (no tables)")
            
            conn.close()
            print("\n✅ Connection test passed!")
            return True
            
        except psycopg.OperationalError as e:
            print(f"❌ Connection failed: {e}")
            print("\nCommon causes:")
            print("  - Wrong hostname: Check Render dashboard")
            print("  - Wrong credentials: Verify user/password")
            print("  - Database not running: Check Render status")
            print("  - Network/firewall: Render databases are public by default")
            return False
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def table_statistics(self) -> bool:
        """Show statistics about the clinical_trials table"""
        print("\n" + "="*80)
        print("TABLE STATISTICS")
        print("="*80)
        
        if not PSYCOPG_AVAILABLE:
            print("❌ psycopg is not installed")
            return False
        
        try:
            conn = psycopg.connect(
                host=self.config["RENDER_DB_HOST"],
                port=int(self.config["RENDER_DB_PORT"]),
                dbname=self.config["RENDER_DB_NAME"],
                user=self.config["RENDER_DB_USER"],
                password=self.config["RENDER_DB_PASSWORD"],
                sslmode="require"
            )
            
            with conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS(
                        SELECT 1 FROM information_schema.tables 
                        WHERE table_schema = 'public' AND table_name = 'clinical_trials'
                    )
                """)
                
                if not cur.fetchone()[0]:
                    print("❌ clinical_trials table does not exist")
                    print("   Run load_to_render_postgres.py with CREATE_TABLE = True")
                    conn.close()
                    return False
                
                # Row count
                cur.execute("SELECT COUNT(*) FROM clinical_trials")
                row_count = cur.fetchone()[0]
                print(f"\n📊 Record Count: {row_count:,}")
                
                # Table size
                cur.execute("""
                    SELECT pg_size_pretty(pg_total_relation_size('clinical_trials'))
                """)
                table_size = cur.fetchone()[0]
                print(f"📦 Table Size: {table_size}")
                
                # Sample data
                print(f"\n📋 Sample Records (first 5):")
                cur.execute("""
                    SELECT nct_id, brief_title, overall_status, start_date 
                    FROM clinical_trials 
                    LIMIT 5
                """)
                samples = cur.fetchall()
                
                if samples:
                    for i, (nct_id, title, status, date) in enumerate(samples, 1):
                        print(f"\n   {i}. NCT ID: {nct_id}")
                        print(f"      Title: {title[:60]}...")
                        print(f"      Status: {status}")
                        print(f"      Start: {date}")
                else:
                    print("   (no records)")
                
                # Columns statistics
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_schema = 'public' AND table_name = 'clinical_trials'
                    ORDER BY ordinal_position
                """)
                columns = cur.fetchall()
                print(f"\n📋 Column Count: {len(columns)}")
                print("   Columns:")
                for col_name, col_type in columns[:10]:  # Show first 10
                    print(f"     - {col_name}: {col_type}")
                if len(columns) > 10:
                    print(f"     ... and {len(columns) - 10} more")
                
                # Embedding statistics
                cur.execute("""
                    SELECT COUNT(*) FROM clinical_trials 
                    WHERE eligibility_criteria_embedding IS NOT NULL
                """)
                embedding_count = cur.fetchone()[0]
                if embedding_count > 0:
                    embedding_pct = (embedding_count / row_count) * 100 if row_count > 0 else 0
                    print(f"\n🧠 Embeddings: {embedding_count:,} ({embedding_pct:.1f}%)")
                else:
                    print(f"\n🧠 Embeddings: None generated yet")
            
            conn.close()
            print("\n✅ Statistics retrieved successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def create_table(self) -> bool:
        """Create the clinical_trials table"""
        print("\n" + "="*80)
        print("CREATE TABLE")
        print("="*80)
        
        if not PSYCOPG_AVAILABLE:
            print("❌ psycopg is not installed")
            return False
        
        try:
            conn = psycopg.connect(
                host=self.config["RENDER_DB_HOST"],
                port=int(self.config["RENDER_DB_PORT"]),
                dbname=self.config["RENDER_DB_NAME"],
                user=self.config["RENDER_DB_USER"],
                password=self.config["RENDER_DB_PASSWORD"],
                sslmode="require"
            )
            
            with conn.cursor() as cur:
                print("\n📋 Creating pgvector extension...")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                print("📋 Creating clinical_trials table...")
                cur.execute("""
                CREATE TABLE IF NOT EXISTS clinical_trials (
                    id SERIAL PRIMARY KEY,
                    nct_id TEXT UNIQUE,
                    org_study_id TEXT,
                    brief_title TEXT,
                    official_title TEXT,
                    overall_status TEXT,
                    start_date DATE,
                    start_date_type TEXT,
                    primary_completion_date DATE,
                    primary_completion_date_type TEXT,
                    completion_date DATE,
                    completion_date_type TEXT,
                    study_first_submit_date DATE,
                    last_update_date DATE,
                    last_update_date_type TEXT,
                    lead_sponsor_name TEXT,
                    lead_sponsor_class TEXT,
                    collaborators JSONB,
                    arm_groups JSONB,
                    primary_outcomes JSONB,
                    secondary_outcomes JSONB,
                    brief_summary TEXT,
                    detailed_description TEXT,
                    conditions JSONB,
                    study_type TEXT,
                    healthy_volunteers BOOLEAN,
                    phases JSONB,
                    allocation TEXT,
                    enrollment_count FLOAT,
                    eligibility_criteria TEXT,
                    eligibility_criteria_embedding VECTOR(1536),
                    gender_based BOOLEAN,
                    gender_description TEXT,
                    sex TEXT,
                    minimum_age FLOAT,
                    maximum_age FLOAT,
                    std_ages JSONB,
                    study_population TEXT,
                    sampling_method TEXT,
                    study_references JSONB,
                    see_also_links JSONB,
                    avail_ipds JSONB,
                    drug_name TEXT,
                    drug_description TEXT,
                    has_results BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                conn.commit()
                print("\n✅ Table created successfully!")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False
    
    def drop_table(self) -> bool:
        """Drop the clinical_trials table (with confirmation)"""
        print("\n" + "="*80)
        print("DROP TABLE")
        print("="*80)
        
        response = input("\n⚠️  WARNING: This will delete all data in clinical_trials table!\nAre you sure? (yes/no): ")
        
        if response.lower() not in ['yes', 'y']:
            print("❌ Cancelled")
            return False
        
        if not PSYCOPG_AVAILABLE:
            print("❌ psycopg is not installed")
            return False
        
        try:
            conn = psycopg.connect(
                host=self.config["RENDER_DB_HOST"],
                port=int(self.config["RENDER_DB_PORT"]),
                dbname=self.config["RENDER_DB_NAME"],
                user=self.config["RENDER_DB_USER"],
                password=self.config["RENDER_DB_PASSWORD"],
                sslmode="require"
            )
            
            with conn.cursor() as cur:
                print("\n🗑️  Dropping table...")
                cur.execute("DROP TABLE IF EXISTS clinical_trials")
                conn.commit()
                print("✅ Table dropped successfully!")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Render PostgreSQL Configuration and Testing Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_render_config.py --check-env
  python test_render_config.py --test-connection
  python test_render_config.py --table-stats
  python test_render_config.py --create-table
  python test_render_config.py --drop-table
        """
    )
    
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check if environment variables are set correctly"
    )
    
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test connection to the database"
    )
    
    parser.add_argument(
        "--table-stats",
        action="store_true",
        help="Show statistics about the clinical_trials table"
    )
    
    parser.add_argument(
        "--create-table",
        action="store_true",
        help="Create the clinical_trials table"
    )
    
    parser.add_argument(
        "--drop-table",
        action="store_true",
        help="Drop the clinical_trials table (requires confirmation)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all checks and tests"
    )
    
    args = parser.parse_args()
    
    tester = RenderConfigTester()
    
    # If no arguments, show help
    if not any([args.check_env, args.test_connection, args.table_stats, 
                args.create_table, args.drop_table, args.all]):
        parser.print_help()
        sys.exit(1)
    
    # Run requested tests
    success = True
    
    if args.all or args.check_env:
        if not tester.check_environment():
            success = False
    
    if args.all or args.test_connection:
        if not tester.test_connection():
            success = False
    
    if args.all or args.table_stats:
        if not tester.table_statistics():
            success = False
    
    if args.create_table:
        if not tester.create_table():
            success = False
    
    if args.drop_table:
        if not tester.drop_table():
            success = False
    
    # Summary
    print("\n" + "="*80)
    if success:
        print("✅ All checks passed!")
    else:
        print("❌ Some checks failed. See above for details.")
    print("="*80 + "\n")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
