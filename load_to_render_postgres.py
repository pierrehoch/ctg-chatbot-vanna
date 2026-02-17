"""
Clinical Trials Data Pipeline for Render PostgreSQL Database

This script fetches clinical trials data from ClinicalTrials.gov API,
processes and prepares it, generates embeddings, and loads it into a
PostgreSQL database on Render.

Author: Data Pipeline Team
Date: December 2024
"""

import pandas as pd
import requests
import json
import os
import sys
import psycopg
from psycopg import sql
import numpy as np
import datetime
import math
import time
import argparse
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, environment variables must be set manually
    pass

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from src.utils.get_embedding import get_openai_embedding, get_openai_embeddings_batch
    from src.utils.logging_config import setup_logging, get_logger
except ImportError:
    try:
        from utils.get_embedding import get_openai_embedding, get_openai_embeddings_batch
        from utils.logging_config import setup_logging, get_logger
    except ImportError:
        from get_embedding import get_openai_embedding, get_openai_embeddings_batch
        from logging_config import setup_logging, get_logger


@dataclass
class RenderDatabaseConfig:
    """Configuration for Render PostgreSQL database connection"""
    database_url: str
    ssl_mode: str = "require"
    
    @classmethod
    def from_components(cls, host: str, user: str, password: str, dbname: str = "", 
                       port: int = 5432, ssl_mode: str = "require") -> 'RenderDatabaseConfig':
        """Create config from individual components (for backwards compatibility)"""
        # Build URL from components
        database_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        return cls(database_url=database_url, ssl_mode=ssl_mode)
    
    def to_connection_kwargs(self) -> Dict[str, Any]:
        """Convert config to psycopg connection kwargs"""
        return {
            "conninfo": self.database_url,
            "sslmode": self.ssl_mode,
        }


@dataclass
class CheckpointMetadata:
    """Metadata for pipeline checkpoints"""
    step: str  # 'fetch', 'prepare', 'embed', 'load'
    timestamp: str
    num_records: int
    filter_by_start_date: bool
    start_year: Optional[int]
    file_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "step": self.step,
            "timestamp": self.timestamp,
            "num_records": self.num_records,
            "filter_by_start_date": self.filter_by_start_date,
            "start_year": self.start_year,
            "file_path": self.file_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create from dictionary"""
        return cls(
            step=data["step"],
            timestamp=data["timestamp"],
            num_records=data["num_records"],
            filter_by_start_date=data["filter_by_start_date"],
            start_year=data.get("start_year"),
            file_path=data["file_path"]
        )


class CheckpointManager:
    """Manage checkpoints for the pipeline"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpoint manager"""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        
    def save_checkpoint(self, df: pd.DataFrame, step: str, metadata: Dict[str, Any]) -> str:
        """
        Save a checkpoint with metadata
        
        Args:
            df: DataFrame to save
            step: Step name ('fetch', 'prepare', 'embed')
            metadata: Additional metadata
            
        Returns:
            str: Path to saved checkpoint file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{step}_{timestamp}.parquet"
        filepath = self.checkpoint_dir / filename
        
        # Save DataFrame
        df.to_parquet(filepath, index=False)
        
        # Create checkpoint metadata
        checkpoint_meta = CheckpointMetadata(
            step=step,
            timestamp=timestamp,
            num_records=len(df),
            filter_by_start_date=metadata.get("filter_by_start_date", False),
            start_year=metadata.get("start_year"),
            file_path=str(filepath)
        )
        
        # Update metadata file
        self._update_metadata(checkpoint_meta)
        
        return str(filepath)
    
    def load_checkpoint(self, step: str) -> Optional[Tuple[pd.DataFrame, CheckpointMetadata]]:
        """
        Load the latest checkpoint for a given step
        
        Args:
            step: Step name to load
            
        Returns:
            Tuple of (DataFrame, metadata) or None if no checkpoint exists
        """
        metadata = self.get_latest_metadata(step)
        if metadata is None:
            return None
        
        try:
            df = pd.read_parquet(metadata.file_path)
            return df, metadata
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}")
            return None
    
    def get_latest_metadata(self, step: str) -> Optional[CheckpointMetadata]:
        """Get metadata for the latest checkpoint of a given step"""
        all_metadata = self._load_all_metadata()
        step_metadata = [m for m in all_metadata if m.step == step]
        
        if not step_metadata:
            return None
        
        # Sort by timestamp and return latest
        step_metadata.sort(key=lambda x: x.timestamp, reverse=True)
        return step_metadata[0]
    
    def _load_all_metadata(self) -> List[CheckpointMetadata]:
        """Load all checkpoint metadata"""
        if not self.metadata_file.exists():
            return []
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return [CheckpointMetadata.from_dict(item) for item in data]
        except Exception:
            return []
    
    def _update_metadata(self, new_metadata: CheckpointMetadata):
        """Add or update checkpoint metadata"""
        all_metadata = self._load_all_metadata()
        all_metadata.append(new_metadata)
        
        # Save updated metadata
        with open(self.metadata_file, 'w') as f:
            json.dump([m.to_dict() for m in all_metadata], f, indent=2)
    
    def list_checkpoints(self) -> Dict[str, List[CheckpointMetadata]]:
        """List all checkpoints grouped by step"""
        all_metadata = self._load_all_metadata()
        
        grouped = {}
        for meta in all_metadata:
            if meta.step not in grouped:
                grouped[meta.step] = []
            grouped[meta.step].append(meta)
        
        # Sort each group by timestamp
        for step in grouped:
            grouped[step].sort(key=lambda x: x.timestamp, reverse=True)
        
        return grouped
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3):
        """Remove old checkpoint files, keeping only the last N for each step"""
        grouped = self.list_checkpoints()
        
        for step, checkpoints in grouped.items():
            if len(checkpoints) > keep_last_n:
                to_remove = checkpoints[keep_last_n:]
                for checkpoint in to_remove:
                    try:
                        filepath = Path(checkpoint.file_path)
                        if filepath.exists():
                            filepath.unlink()
                            print(f"🗑️  Removed old checkpoint: {filepath.name}")
                    except Exception as e:
                        print(f"⚠️  Error removing {checkpoint.file_path}: {e}")


class ClinicalTrialsDataPipeline:
    """
    Complete pipeline for fetching, processing, and loading clinical trials data
    to a Render PostgreSQL database
    """
    
    def __init__(self, db_config: RenderDatabaseConfig, logger=None, checkpoint_dir: str = "checkpoints"):
        """
        Initialize the pipeline with database configuration
        
        Args:
            db_config: RenderDatabaseConfig instance with database credentials
            logger: Optional logger instance (will create if not provided)
            checkpoint_dir: Directory for storing checkpoints
        """
        self.db_config = db_config
        self.logger = logger if logger is not None else get_logger()
        self.conn = None
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
    def get_db_connection(self) -> psycopg.Connection:
        """
        Create and return a PostgreSQL connection to Render database
        
        Returns:
            psycopg.Connection: Database connection object
        """
        try:
            self.logger.info("Connecting to Render PostgreSQL database...")
            self.conn = psycopg.connect(**self.db_config.to_connection_kwargs())
            self.logger.info("✅ Successfully connected to Render PostgreSQL database")
            return self.conn
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to database: {e}")
            raise
    
    def close_connection(self):
        """Close the database connection safely"""
        if self.conn is not None:
            try:
                self.conn.close()
                self.logger.info("Database connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing database connection: {e}")
    
    def fetch_clinical_trials(self, total_limit: int = 10, filter_by_start_date: bool = False, start_year: int = 2025) -> Optional[pd.DataFrame]:
        """
        Fetch clinical trials from ClinicalTrials.gov API with pagination support
        
        Args:
            total_limit: Total number of records to fetch across all pages
            filter_by_start_date: Whether to filter by start date in the API call
            start_year: Year to filter studies that started after this year (exclusive)
            
        Returns:
            pd.DataFrame: DataFrame containing clinical trials data
        """
        filter_desc = f" (start date after {start_year})" if filter_by_start_date else ""
        self.logger.info(f"📥 Starting to fetch clinical trials (limit: {total_limit}){filter_desc}...")
        
        base_url = "https://clinicaltrials.gov/api/v2/studies"
        
        params = {
            "format": "json",
            "pageSize": min(total_limit, 2000),
        }
        
        # Add start date filter using the API's advanced filter syntax
        if filter_by_start_date:
            # Use AREA[StartDate]RANGE to filter studies with start date after the specified year
            # The API uses YYYY-MM-DD format, so we create a range from the year after start_year to MAX
            filter_start_date = f"{start_year + 1}-01-01"
            params["filter.advanced"] = f"AREA[StartDate]RANGE[{filter_start_date}, MAX]"
            self.logger.info(f"Applying API filter: studies starting from {filter_start_date} onwards")
            print(f"🔍 Filtering studies with start date after {start_year}")
        
        all_studies = []
        records_fetched = 0
        page_num = 0
        
        try:
            while records_fetched < total_limit:
                page_num += 1
                self.logger.debug(f"Fetching page {page_num}...")
                
                response = requests.get(base_url, params=params, timeout=30)
                
                if response.status_code != 200:
                    raise RuntimeError(f"API error {response.status_code}: {response.text}")
                
                data = response.json()
                studies = data.get("studies", [])
                
                all_studies.extend(studies)
                records_fetched += len(studies)
                
                print(f"✅ Fetched {len(studies)} records on page {page_num}, total: {records_fetched}")
                self.logger.info(f"Fetched {len(studies)} records on page {page_num}, total: {records_fetched}")
                
                # Check for next page
                next_page_token = data.get("nextPageToken")
                if not next_page_token or records_fetched >= total_limit:
                    break
                
                # Update pagination token
                params["pageToken"] = next_page_token
                params["pageSize"] = min(total_limit - records_fetched, 2000)
                
                # Rate limiting
                time.sleep(0.5)
            
            # Convert to DataFrame
            if all_studies:
                df = pd.json_normalize(all_studies)
                
                # Define columns to keep
                columns_to_keep = [
                    "protocolSection.identificationModule.nctId",
                    "protocolSection.identificationModule.orgStudyIdInfo.id",
                    "protocolSection.identificationModule.briefTitle",
                    "protocolSection.identificationModule.officialTitle",
                    "protocolSection.statusModule.overallStatus",
                    "protocolSection.statusModule.startDateStruct.date",
                    "protocolSection.statusModule.startDateStruct.type",
                    "protocolSection.statusModule.primaryCompletionDateStruct.date",
                    "protocolSection.statusModule.primaryCompletionDateStruct.type",
                    "protocolSection.statusModule.completionDateStruct.date",
                    "protocolSection.statusModule.completionDateStruct.type",
                    "protocolSection.statusModule.studyFirstSubmitQcDate",
                    "protocolSection.statusModule.lastUpdatePostDateStruct.date",
                    "protocolSection.statusModule.lastUpdatePostDateStruct.type",
                    "protocolSection.sponsorCollaboratorsModule.leadSponsor.name",
                    "protocolSection.sponsorCollaboratorsModule.leadSponsor.class",
                    "protocolSection.sponsorCollaboratorsModule.collaborators",
                    "protocolSection.armsInterventionsModule.interventions",
                    "protocolSection.armsInterventionsModule.armGroups",
                    "protocolSection.outcomesModule.primaryOutcomes",
                    "protocolSection.outcomesModule.secondaryOutcomes",
                    "protocolSection.descriptionModule.briefSummary",
                    "protocolSection.descriptionModule.detailedDescription",
                    "protocolSection.conditionsModule.conditions",
                    "protocolSection.designModule.studyType",
                    "protocolSection.designModule.phases",
                    "protocolSection.designModule.designInfo.allocation",
                    "protocolSection.designModule.enrollmentInfo.count",
                    "protocolSection.eligibilityModule.eligibilityCriteria",
                    "protocolSection.eligibilityModule.healthyVolunteers",
                    "protocolSection.eligibilityModule.genderBased",
                    "protocolSection.eligibilityModule.genderDescription",
                    "protocolSection.eligibilityModule.sex",
                    "protocolSection.eligibilityModule.minimumAge",
                    "protocolSection.eligibilityModule.maximumAge",
                    "protocolSection.eligibilityModule.stdAges",
                    "protocolSection.eligibilityModule.studyPopulation",
                    "protocolSection.eligibilityModule.samplingMethod",
                    "protocolSection.referencesModule.references",
                    "protocolSection.referencesModule.seeAlsoLinks",
                    "protocolSection.referencesModule.availIpds",
                    "hasResults"
                ]
                
                # Filter to existing columns
                existing_columns = [col for col in columns_to_keep if col in df.columns]
                filtered_df = df[existing_columns]
                
                self.logger.info(f"✅ Fetched {len(filtered_df)} records with {len(existing_columns)} columns")
                print(f"✅ Successfully fetched {len(filtered_df)} clinical trials")
                return filtered_df
            else:
                self.logger.warning("No studies were fetched from API")
                print("⚠️  No studies were fetched")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error fetching clinical trials: {e}")
            print(f"❌ Error fetching data: {e}")
            raise
    
    def get_existing_nct_ids(self, conn: psycopg.Connection) -> set:
        """
        Get set of NCT IDs that already exist in the database
        
        Args:
            conn: Database connection
            
        Returns:
            set: Set of existing NCT IDs
        """
        try:
            with conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'clinical_trials'
                    )
                """)
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    self.logger.info("Table doesn't exist yet, no existing records")
                    return set()
                
                # Get all existing NCT IDs
                cur.execute("SELECT nct_id FROM clinical_trials")
                existing_ids = {row[0] for row in cur.fetchall()}
                
                self.logger.info(f"Found {len(existing_ids)} existing records in database")
                return existing_ids
        except Exception as e:
            self.logger.warning(f"Error fetching existing NCT IDs: {e}")
            return set()
    
    def prepare_data(self, df: pd.DataFrame, existing_nct_ids: Optional[set] = None) -> Optional[pd.DataFrame]:
        """
        Prepare and transform clinical trials data for database insertion
        
        Note: Filtering by start date is now handled by the API in fetch_clinical_trials(),
        so this method no longer needs to perform client-side filtering.
        
        Args:
            df: Raw DataFrame from API
            existing_nct_ids: Optional set of NCT IDs that already exist in database
            
        Returns:
            pd.DataFrame: Prepared data ready for embeddings and database
        """
        self.logger.info("🔄 Preparing clinical trials data...")
        
        df_filtered = df.copy()
        
        # Filter out existing records if NCT IDs are provided
        if existing_nct_ids and len(existing_nct_ids) > 0:
            nct_col = "protocolSection.identificationModule.nctId"
            if nct_col in df_filtered.columns:
                initial_count = len(df_filtered)
                df_filtered = df_filtered[~df_filtered[nct_col].isin(existing_nct_ids)]
                filtered_count = initial_count - len(df_filtered)
                
                self.logger.info(f"Filtered out {filtered_count} existing records, {len(df_filtered)} new records to process")
                print(f"⏭️  Skipping {filtered_count} existing records, processing {len(df_filtered)} new records")
                
                if len(df_filtered) == 0:
                    self.logger.info("All records already exist in database")
                    print("✅ All records already exist in database - nothing to do!")
                    return None
        
        if len(df_filtered) == 0:
            self.logger.warning("No records found")
            return None
        
        # Extract intervention information
        def extract_intervention_info(interventions) -> Tuple[Optional[str], Optional[str]]:
            """Extract intervention names and descriptions"""
            if interventions is None:
                return None, None
            
            try:
                if pd.isna(interventions):
                    return None, None
            except (TypeError, ValueError):
                pass
            
            if isinstance(interventions, str):
                if not interventions.strip():
                    return None, None
                try:
                    interventions = json.loads(interventions)
                except (json.JSONDecodeError, TypeError):
                    return None, None
            
            if not isinstance(interventions, list):
                return None, None
            
            names, descriptions = [], []
            for intervention in interventions:
                if isinstance(intervention, dict):
                    int_type = intervention.get('type', 'UNKNOWN')
                    name = intervention.get('name')
                    description = intervention.get('description')
                    
                    if name:
                        names.append(f"{int_type}: {name}")
                    if description:
                        descriptions.append(description)
            
            return (
                '; '.join(names) if names else None,
                '; '.join(descriptions) if descriptions else None
            )
        
        # Apply intervention extraction
        if 'protocolSection.armsInterventionsModule.interventions' in df_filtered.columns:
            intervention_info = df_filtered['protocolSection.armsInterventionsModule.interventions'].apply(extract_intervention_info)
            df_filtered['drug_name'] = [info[0] for info in intervention_info]
            df_filtered['drug_description'] = [info[1] for info in intervention_info]
        
        # Column mapping
        column_mapping = {
            "protocolSection.identificationModule.nctId": "nct_id",
            "protocolSection.identificationModule.orgStudyIdInfo.id": "org_study_id",
            "protocolSection.identificationModule.briefTitle": "brief_title",
            "protocolSection.identificationModule.officialTitle": "official_title",
            "protocolSection.statusModule.overallStatus": "overall_status",
            "protocolSection.statusModule.startDateStruct.date": "start_date",
            "protocolSection.statusModule.startDateStruct.type": "start_date_type",
            "protocolSection.statusModule.primaryCompletionDateStruct.date": "primary_completion_date",
            "protocolSection.statusModule.primaryCompletionDateStruct.type": "primary_completion_date_type",
            "protocolSection.statusModule.completionDateStruct.date": "completion_date",
            "protocolSection.statusModule.completionDateStruct.type": "completion_date_type",
            "protocolSection.statusModule.studyFirstSubmitQcDate": "study_first_submit_date",
            "protocolSection.statusModule.lastUpdatePostDateStruct.date": "last_update_date",
            "protocolSection.statusModule.lastUpdatePostDateStruct.type": "last_update_date_type",
            "protocolSection.sponsorCollaboratorsModule.leadSponsor.name": "lead_sponsor_name",
            "protocolSection.sponsorCollaboratorsModule.leadSponsor.class": "lead_sponsor_class",
            "protocolSection.sponsorCollaboratorsModule.collaborators": "collaborators",
            "protocolSection.armsInterventionsModule.armGroups": "arm_groups",
            "protocolSection.outcomesModule.primaryOutcomes": "primary_outcomes",
            "protocolSection.outcomesModule.secondaryOutcomes": "secondary_outcomes",
            "protocolSection.descriptionModule.briefSummary": "brief_summary",
            "protocolSection.descriptionModule.detailedDescription": "detailed_description",
            "protocolSection.conditionsModule.conditions": "conditions",
            "protocolSection.designModule.studyType": "study_type",
            "protocolSection.designModule.phases": "phases",
            "protocolSection.designModule.designInfo.allocation": "allocation",
            "protocolSection.designModule.enrollmentInfo.count": "enrollment_count",
            "protocolSection.eligibilityModule.eligibilityCriteria": "eligibility_criteria",
            "protocolSection.eligibilityModule.healthyVolunteers": "healthy_volunteers",
            "protocolSection.eligibilityModule.genderBased": "gender_based",
            "protocolSection.eligibilityModule.genderDescription": "gender_description",
            "protocolSection.eligibilityModule.sex": "sex",
            "protocolSection.eligibilityModule.minimumAge": "minimum_age",
            "protocolSection.eligibilityModule.maximumAge": "maximum_age",
            "protocolSection.eligibilityModule.stdAges": "std_ages",
            "protocolSection.eligibilityModule.studyPopulation": "study_population",
            "protocolSection.eligibilityModule.samplingMethod": "sampling_method",
            "protocolSection.referencesModule.references": "study_references",
            "protocolSection.referencesModule.seeAlsoLinks": "see_also_links",
            "protocolSection.referencesModule.availIpds": "avail_ipds",
            "hasResults": "has_results",
            "drug_name": "drug_name",
            "drug_description": "drug_description"
        }
        
        # Rename columns
        renamed_df = df_filtered.copy()
        for df_col, db_col in column_mapping.items():
            if df_col in renamed_df.columns:
                renamed_df[db_col] = renamed_df[df_col]
        
        # JSONB columns requiring special handling
        jsonb_columns = [
            "collaborators", "conditions", "phases", "std_ages",
            "study_references", "see_also_links", "avail_ipds",
            "arm_groups", "primary_outcomes", "secondary_outcomes"
        ]
        
        def safe_json_serialize(obj):
            """Safely serialize objects to JSON-compatible format"""
            if obj is None:
                return None
            
            if isinstance(obj, np.ndarray):
                return [safe_json_serialize(item) for item in obj.tolist()]
            
            if isinstance(obj, (list, tuple)):
                return [safe_json_serialize(item) for item in obj]
            
            if isinstance(obj, dict):
                return {k: safe_json_serialize(v) for k, v in obj.items()}
            
            try:
                if pd.isna(obj):
                    return None
            except (TypeError, ValueError):
                pass
            
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.str_):
                return str(obj)
            else:
                return obj
        
        # Process JSONB columns
        for col in jsonb_columns:
            if col in renamed_df.columns:
                def process_json_column(x):
                    if x is None:
                        return None
                    try:
                        serialized = safe_json_serialize(x)
                        return json.dumps(serialized) if serialized is not None else None
                    except Exception as e:
                        self.logger.warning(f"Error serializing {col}: {e}")
                        return json.dumps(str(x))
                
                renamed_df[col] = renamed_df[col].apply(process_json_column)
        
        # Select database columns
        db_columns = [
            "nct_id", "org_study_id", "brief_title", "official_title", "overall_status",
            "start_date", "start_date_type", "primary_completion_date", "primary_completion_date_type",
            "completion_date", "completion_date_type", "study_first_submit_date",
            "last_update_date", "last_update_date_type", "lead_sponsor_name",
            "lead_sponsor_class", "collaborators", "arm_groups", "primary_outcomes", "secondary_outcomes",
            "brief_summary", "detailed_description", "conditions", "study_type", "phases", "allocation",
            "enrollment_count", "eligibility_criteria", "healthy_volunteers", "gender_based",
            "gender_description", "sex", "minimum_age", "maximum_age", "std_ages", "study_population",
            "sampling_method", "study_references", "see_also_links", "avail_ipds",
            "drug_name", "drug_description", "has_results"
        ]
        
        cols_to_use = [col for col in db_columns if col in renamed_df.columns]
        final_df = renamed_df[cols_to_use].copy()
        
        # Convert value types
        def convert_value(value):
            """Convert numpy types to Python native types"""
            if value is None:
                return None
            elif isinstance(value, float) and math.isnan(value):
                return None
            elif pd.isna(value):
                return None
            elif isinstance(value, (np.integer, np.int64)):
                return int(value)
            elif isinstance(value, (np.floating, np.float64)):
                return int(value) if value.is_integer() else float(value)
            elif isinstance(value, np.bool_):
                return bool(value)
            elif isinstance(value, np.str_):
                return str(value)
            elif isinstance(value, float):
                return int(value) if value.is_integer() else value
            elif isinstance(value, (int, bool, str)):
                return value
            else:
                return None
        
        # Apply conversions
        for col in final_df.columns:
            final_df[col] = final_df[col].apply(convert_value)
        
        # Parse dates
        def parse_date(date_str):
            """Convert YYYY-MM-DD, YYYY-MM, or YYYY to datetime.date"""
            if pd.isna(date_str) or not date_str:
                return None
            try:
                if isinstance(date_str, datetime.date):
                    return date_str
                date_str = str(date_str)
                if len(date_str) == 10:
                    return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
                elif len(date_str) == 7:
                    return datetime.datetime.strptime(date_str + "-01", "%Y-%m-%d").date()
                elif len(date_str) == 4:
                    return datetime.datetime.strptime(date_str + "-01-01", "%Y-%m-%d").date()
            except Exception:
                pass
            return None
        
        date_columns = [
            "start_date", "primary_completion_date", "completion_date",
            "study_first_submit_date", "last_update_date"
        ]
        for col in date_columns:
            if col in final_df.columns:
                final_df[col] = final_df[col].apply(parse_date)
        
        # Parse ages
        def parse_age(age_str):
            """Convert age strings to integer values"""
            if pd.isna(age_str) or not isinstance(age_str, str):
                return None
            
            age_str = age_str.rstrip('.').lower()
            if age_str in ('n/a', '', 'na'):
                return None
            
            try:
                number = ''.join(c for c in age_str if c.isdigit() or c == '.')
                if not number:
                    return None
                
                value = float(number)
                if 'month' in age_str:
                    value = int(value / 12)
                elif 'week' in age_str:
                    value = int(value / 52)
                elif 'day' in age_str:
                    value = int(value / 365)
                else:
                    value = int(value)
                
                return value
            except Exception:
                return None
        
        age_columns = ["minimum_age", "maximum_age"]
        for col in age_columns:
            if col in final_df.columns:
                final_df[col] = final_df[col].apply(parse_age)
        
        self.logger.info(f"✅ Prepared {len(final_df)} records for database insertion")
        return final_df
    
    def generate_embeddings(self, df: pd.DataFrame, batch_size: int = 500) -> Tuple[pd.DataFrame, bool]:
        """
        Generate embeddings for eligibility criteria
        
        Args:
            df: DataFrame with prepared data
            batch_size: Batch size for embedding generation
            
        Returns:
            tuple: (DataFrame with embeddings, bool indicating success)
        """
        self.logger.info("🧠 Generating embeddings for eligibility criteria...")
        
        if df is None or len(df) == 0:
            self.logger.warning("No data provided for embedding generation")
            return df, False
        
        df_with_embeddings = df.copy()
        
        if "eligibility_criteria" not in df_with_embeddings.columns:
            self.logger.warning("No eligibility_criteria column found")
            df_with_embeddings["eligibility_criteria_embedding"] = None
            return df_with_embeddings, False
        
        print("🧠 Generating embeddings...")
        
        # Prepare texts
        texts_to_embed = []
        for text in df_with_embeddings["eligibility_criteria"]:
            if pd.isna(text) or text is None or str(text).strip() == "":
                texts_to_embed.append(None)
            else:
                texts_to_embed.append(str(text).strip())
        
        total_records = len(df_with_embeddings)
        valid_records = sum(1 for t in texts_to_embed if t is not None)
        self.logger.info(f"Found {valid_records}/{total_records} records with eligibility criteria")
        
        if valid_records > 0:
            try:
                embeddings = get_openai_embeddings_batch(texts_to_embed, batch_size=batch_size)
                df_with_embeddings["eligibility_criteria_embedding"] = embeddings
                
                non_null_embeddings = sum(1 for e in embeddings if e is not None)
                success_rate = (non_null_embeddings / total_records) * 100
                
                self.logger.info(f"✅ Generated {non_null_embeddings}/{total_records} embeddings ({success_rate:.1f}%)")
                print(f"✅ Generated {non_null_embeddings}/{total_records} embeddings")
                
                return df_with_embeddings, non_null_embeddings > 0
            except Exception as e:
                self.logger.error(f"Error generating embeddings: {e}")
                df_with_embeddings["eligibility_criteria_embedding"] = None
                return df_with_embeddings, False
        else:
            df_with_embeddings["eligibility_criteria_embedding"] = None
            return df_with_embeddings, False
    
    def create_table(self, conn: psycopg.Connection):
        """
        Create the clinical_trials table in PostgreSQL
        
        Args:
            conn: Database connection
        """
        self.logger.info("📋 Creating clinical_trials table...")
        
        try:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
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
                self.logger.info("✅ Created clinical_trials table")
                print("✅ Table created successfully")
        except Exception as e:
            self.logger.error(f"Error creating table: {e}")
            raise
    
    def drop_table(self, conn: psycopg.Connection):
        """
        Drop the clinical_trials table
        
        Args:
            conn: Database connection
        """
        self.logger.info("Dropping clinical_trials table...")
        
        try:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS clinical_trials")
                conn.commit()
                self.logger.info("✅ Dropped clinical_trials table")
                print("✅ Table dropped")
        except Exception as e:
            self.logger.error(f"Error dropping table: {e}")
            raise
    
    def load_data(
        self,
        conn: psycopg.Connection,
        df: pd.DataFrame,
        embedding_generated: bool = False
    ) -> int:
        """
        Load prepared data into PostgreSQL database
        
        Args:
            conn: Database connection
            df: DataFrame with data to load
            embedding_generated: Whether embeddings were generated
            
        Returns:
            int: Number of records inserted
        """
        self.logger.info(f"💾 Loading {len(df)} records to database...")
        
        if df is None or len(df) == 0:
            self.logger.warning("No data to load")
            return 0
        
        # Define columns
        db_columns = [
            "nct_id", "org_study_id", "brief_title", "official_title", "overall_status",
            "start_date", "start_date_type", "primary_completion_date", "primary_completion_date_type",
            "completion_date", "completion_date_type", "study_first_submit_date",
            "last_update_date", "last_update_date_type", "lead_sponsor_name",
            "lead_sponsor_class", "collaborators", "arm_groups", "primary_outcomes", "secondary_outcomes",
            "brief_summary", "detailed_description", "conditions", "study_type", "phases", "allocation",
            "enrollment_count", "eligibility_criteria", "healthy_volunteers", "gender_based",
            "gender_description", "sex", "minimum_age", "maximum_age", "std_ages", "study_population",
            "sampling_method", "study_references", "see_also_links", "avail_ipds",
            "drug_name", "drug_description", "has_results"
        ]
        
        if embedding_generated:
            db_columns.append("eligibility_criteria_embedding")
        
        cols_to_use = [col for col in db_columns if col in df.columns]
        
        # Convert DataFrame to records
        records = df.to_dict('records')
        
        # Format embeddings for PostgreSQL VECTOR type
        for record in records:
            if "eligibility_criteria_embedding" in record:
                embedding = record["eligibility_criteria_embedding"]
                if embedding is not None and isinstance(embedding, list):
                    try:
                        record["eligibility_criteria_embedding"] = f"[{','.join(map(str, embedding))}]"
                    except Exception as e:
                        self.logger.warning(f"Error formatting embedding: {e}")
                        record["eligibility_criteria_embedding"] = None
                else:
                    record["eligibility_criteria_embedding"] = None
        
        # Verify connection
        try:
            with conn.cursor() as test_cur:
                test_cur.execute("SELECT 1")
                test_cur.fetchone()
            self.logger.info("Database connection verified")
        except Exception as e:
            self.logger.error(f"Connection lost: {e}")
            raise
        
        # Insert data in batches
        batch_size = 20 if embedding_generated else 50
        total_records = len(records)
        inserted_count = 0
        failed_records = []
        
        self.logger.info(f"Starting insertion with batch size {batch_size}")
        
        with conn.cursor() as cur:
            cur.execute("SET statement_timeout = '100min'")
            
            columns_str = ', '.join(cols_to_use)
            placeholders = ', '.join(['%s'] * len(cols_to_use))
            
            # Use DO NOTHING to skip existing records (faster and avoids unnecessary updates)
            upsert_query = f"""
            INSERT INTO clinical_trials ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT (nct_id)
            DO NOTHING
            """
            
            for batch_num, batch_start in enumerate(range(0, total_records, batch_size), 1):
                batch_end = min(batch_start + batch_size, total_records)
                batch_records = records[batch_start:batch_end]
                
                batch_values = []
                for record in batch_records:
                    values = [record.get(col) for col in cols_to_use]
                    batch_values.append(values)
                
                success = False
                retry_count = 0
                max_retries = 3
                
                while not success and retry_count < max_retries:
                    try:
                        conn.rollback()
                        cur.executemany(upsert_query, batch_values)
                        conn.commit()
                        
                        inserted_count += len(batch_records)
                        success = True
                        
                        progress = (inserted_count / total_records) * 100
                        self.logger.info(
                            f"✅ Batch {batch_num}: {len(batch_records)} records "
                            f"({inserted_count}/{total_records}, {progress:.1f}%)"
                        )
                        print(f"✅ Batch {batch_num}: {inserted_count}/{total_records} ({progress:.1f}%)")
                        
                    except Exception as e:
                        retry_count += 1
                        error_msg = str(e).lower()
                        
                        if "timeout" in error_msg:
                            self.logger.warning(f"Timeout in batch {batch_num}, reducing batch size")
                            batch_size = max(5, batch_size // 2)
                        else:
                            self.logger.error(f"Error in batch {batch_num}: {e}")
                        
                        if retry_count < max_retries:
                            wait_time = min(30 * retry_count, 120)
                            self.logger.info(f"Retrying in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            # Try individual inserts
                            self.logger.warning("Batch failed, trying individual records")
                            conn.rollback()
                            
                            for record in batch_records:
                                values = [record.get(col) for col in cols_to_use]
                                try:
                                    cur.execute(upsert_query, values)
                                    conn.commit()
                                    inserted_count += 1
                                except Exception as record_e:
                                    self.logger.error(f"Failed to insert {record.get('nct_id')}: {record_e}")
                                    failed_records.append(record.get('nct_id'))
                                    conn.rollback()
                            
                            success = True
        
        if failed_records:
            self.logger.warning(f"Failed to insert {len(failed_records)} records")
            print(f"⚠️  {len(failed_records)} records failed to insert")
        
        success_rate = (inserted_count / total_records) * 100
        self.logger.info(f"✅ Loaded {inserted_count}/{total_records} records ({success_rate:.1f}%)")
        print(f"✅ Successfully loaded {inserted_count}/{total_records} records")
        
        return inserted_count
    
    def run_full_pipeline(
        self,
        num_records: int = 10,
        generate_embeddings: bool = False,
        create_table: bool = False,
        drop_table_first: bool = False,
        filter_by_start_date: bool = False,
        start_year: int = 2025,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline from fetch to load
        
        Args:
            num_records: Number of records to fetch
            generate_embeddings: Whether to generate embeddings
            create_table: Whether to create the table
            drop_table_first: Whether to drop the table first
            filter_by_start_date: Whether to filter by start date (applied in API call)
            start_year: Year to filter studies that started after this year (exclusive)
            skip_existing: Whether to skip records that already exist in the database
            
        Returns:
            dict: Summary of execution
        """
        print("\n" + "="*80)
        print("CLINICAL TRIALS DATA PIPELINE")
        print("="*80)
        
        results = {
            "fetched": 0,
            "existing_skipped": 0,
            "prepared": 0,
            "embeddings_generated": False,
            "table_created": False,
            "loaded": 0
        }
        
        try:
            # Get database connection
            conn = self.get_db_connection()
            
            # Get existing NCT IDs if skip_existing is enabled
            existing_nct_ids = set()
            if skip_existing and not drop_table_first:
                print("\n🔍 Checking for existing records...")
                existing_nct_ids = self.get_existing_nct_ids(conn)
                if len(existing_nct_ids) > 0:
                    print(f"✅ Found {len(existing_nct_ids)} existing records in database\n")
                else:
                    print("✅ No existing records found\n")
            
            # Fetch data
            print("\n📥 STEP 1: Fetching data...")
            df = self.fetch_clinical_trials(num_records, filter_by_start_date, start_year)
            if df is not None:
                results["fetched"] = len(df)
                print(f"✅ Fetched {results['fetched']} records\n")
            else:
                print("❌ Failed to fetch data\n")
                return results
            
            # Prepare data
            print("🔄 STEP 2: Preparing data...")
            prepared_df = self.prepare_data(df, existing_nct_ids if skip_existing else None)
            if prepared_df is not None:
                results["prepared"] = len(prepared_df)
                results["existing_skipped"] = results["fetched"] - results["prepared"]
                print(f"✅ Prepared {results['prepared']} records\n")
            else:
                # Check if we filtered everything out
                if skip_existing and len(existing_nct_ids) > 0:
                    results["existing_skipped"] = results["fetched"]
                    print("✅ All fetched records already exist in database - nothing to process\n")
                else:
                    print("❌ Failed to prepare data\n")
                return results
            
            # Generate embeddings if requested
            embeddings_generated = False
            if generate_embeddings:
                print("🧠 STEP 3: Generating embeddings...")
                prepared_df, embeddings_generated = self.generate_embeddings(prepared_df)
                results["embeddings_generated"] = embeddings_generated
                print()
            
            # Database operations
            print("🗄️  STEP 4: Database operations...")
            
            if drop_table_first:
                self.drop_table(conn)
            
            if create_table:
                self.create_table(conn)
                results["table_created"] = True
            
            # Load data
            loaded_count = self.load_data(conn, prepared_df, embeddings_generated)
            results["loaded"] = loaded_count
            
            # Close connection
            self.close_connection()
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            print(f"❌ Pipeline error: {e}")
            if self.conn:
                self.close_connection()
        
        # Print summary
        print("\n" + "="*80)
        print("EXECUTION SUMMARY")
        print("="*80)
        print(f"✅ Fetched:      {results['fetched']} records")
        if results['existing_skipped'] > 0:
            print(f"⏭️  Skipped:      {results['existing_skipped']} existing records")
        print(f"✅ Prepared:     {results['prepared']} records")
        if results['embeddings_generated']:
            print(f"✅ Embeddings:   Generated")
        if results['table_created']:
            print(f"✅ Table:        Created")
        print(f"✅ Loaded:       {results['loaded']} records")
        print("="*80 + "\n")
        
        return results
    
    def run_step(
        self,
        step: str,
        num_records: int = 10,
        generate_embeddings: bool = True,
        create_table: bool = False,
        drop_table_first: bool = False,
        filter_by_start_date: bool = False,
        start_year: int = 2025,
        skip_existing: bool = True,
        save_checkpoint: bool = True
    ) -> Dict[str, Any]:
        """
        Run a single step of the pipeline with checkpoint support
        
        Args:
            step: Step to run ('fetch', 'prepare', 'embed', 'load', 'all')
            num_records: Number of records to fetch (for 'fetch' step)
            generate_embeddings: Whether to generate embeddings
            create_table: Whether to create table (for 'load' step)
            drop_table_first: Whether to drop table first (for 'load' step)
            filter_by_start_date: Filter by start date (for 'fetch' step)
            start_year: Year filter (for 'fetch' step)
            skip_existing: Skip existing records
            save_checkpoint: Whether to save checkpoint after step
            
        Returns:
            dict: Summary of execution
        """
        if step == 'all':
            return self.run_full_pipeline(
                num_records=num_records,
                generate_embeddings=generate_embeddings,
                create_table=create_table,
                drop_table_first=drop_table_first,
                filter_by_start_date=filter_by_start_date,
                start_year=start_year,
                skip_existing=skip_existing
            )
        
        results = {"step": step, "success": False}
        
        try:
            if step == 'fetch':
                results.update(self._run_fetch_step(
                    num_records, filter_by_start_date, start_year, save_checkpoint
                ))
            elif step == 'prepare':
                results.update(self._run_prepare_step(skip_existing, save_checkpoint))
            elif step == 'embed':
                results.update(self._run_embed_step(save_checkpoint))
            elif step == 'load':
                results.update(self._run_load_step(
                    create_table, drop_table_first
                ))
            else:
                print(f"❌ Unknown step: {step}")
                print("   Valid steps: fetch, prepare, embed, load, all")
                results["error"] = f"Unknown step: {step}"
                
        except Exception as e:
            self.logger.error(f"Error in step {step}: {e}")
            print(f"❌ Error in step {step}: {e}")
            results["error"] = str(e)
        
        return results
    
    def _run_fetch_step(
        self,
        num_records: int,
        filter_by_start_date: bool,
        start_year: int,
        save_checkpoint: bool
    ) -> Dict[str, Any]:
        """Run the fetch step"""
        print("\n" + "="*80)
        print("STEP 1: FETCH DATA")
        print("="*80)
        
        df = self.fetch_clinical_trials(num_records, filter_by_start_date, start_year)
        
        if df is None or len(df) == 0:
            print("❌ Failed to fetch data")
            return {"success": False, "fetched": 0}
        
        results = {"success": True, "fetched": len(df)}
        
        if save_checkpoint:
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                df, 
                "fetch",
                {
                    "filter_by_start_date": filter_by_start_date,
                    "start_year": start_year
                }
            )
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            results["checkpoint"] = checkpoint_path
        
        print(f"\n✅ Fetch step completed: {len(df)} records")
        print("="*80 + "\n")
        
        return results
    
    def _run_prepare_step(
        self,
        skip_existing: bool,
        save_checkpoint: bool
    ) -> Dict[str, Any]:
        """Run the prepare step"""
        print("\n" + "="*80)
        print("STEP 2: PREPARE DATA")
        print("="*80)
        
        # Load fetched data from checkpoint
        checkpoint_data = self.checkpoint_manager.load_checkpoint("fetch")
        if checkpoint_data is None:
            print("❌ No fetch checkpoint found. Run 'fetch' step first.")
            return {"success": False, "error": "No fetch checkpoint"}
        
        df, metadata = checkpoint_data
        print(f"📂 Loaded {len(df)} records from fetch checkpoint")
        
        # Get existing NCT IDs if needed
        existing_nct_ids = set()
        if skip_existing:
            conn = self.get_db_connection()
            existing_nct_ids = self.get_existing_nct_ids(conn)
            self.close_connection()
            if len(existing_nct_ids) > 0:
                print(f"🔍 Found {len(existing_nct_ids)} existing records in database")
        
        # Prepare data
        prepared_df = self.prepare_data(df, existing_nct_ids if skip_existing else None)
        
        if prepared_df is None or len(prepared_df) == 0:
            print("❌ No data to prepare (all records may already exist)")
            return {
                "success": True,
                "prepared": 0,
                "skipped": len(df) if skip_existing else 0
            }
        
        results = {
            "success": True,
            "prepared": len(prepared_df),
            "skipped": len(df) - len(prepared_df)
        }
        
        if save_checkpoint:
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                prepared_df,
                "prepare",
                {
                    "filter_by_start_date": metadata.filter_by_start_date,
                    "start_year": metadata.start_year
                }
            )
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            results["checkpoint"] = checkpoint_path
        
        print(f"\n✅ Prepare step completed: {len(prepared_df)} records")
        print("="*80 + "\n")
        
        return results
    
    def _run_embed_step(self, save_checkpoint: bool) -> Dict[str, Any]:
        """Run the embedding generation step"""
        print("\n" + "="*80)
        print("STEP 3: GENERATE EMBEDDINGS")
        print("="*80)
        
        # Load prepared data from checkpoint
        checkpoint_data = self.checkpoint_manager.load_checkpoint("prepare")
        if checkpoint_data is None:
            print("❌ No prepare checkpoint found. Run 'prepare' step first.")
            return {"success": False, "error": "No prepare checkpoint"}
        
        df, metadata = checkpoint_data
        print(f"📂 Loaded {len(df)} records from prepare checkpoint")
        
        # Generate embeddings
        df_with_embeddings, embeddings_generated = self.generate_embeddings(df)
        
        if not embeddings_generated:
            print("⚠️  No embeddings were generated")
            return {
                "success": True,
                "embeddings_generated": False,
                "records": len(df)
            }
        
        results = {
            "success": True,
            "embeddings_generated": True,
            "records": len(df_with_embeddings)
        }
        
        if save_checkpoint:
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                df_with_embeddings,
                "embed",
                {
                    "filter_by_start_date": metadata.filter_by_start_date,
                    "start_year": metadata.start_year
                }
            )
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            results["checkpoint"] = checkpoint_path
        
        print(f"\n✅ Embed step completed")
        print("="*80 + "\n")
        
        return results
    
    def _run_load_step(
        self,
        create_table: bool,
        drop_table_first: bool
    ) -> Dict[str, Any]:
        """Run the database load step"""
        print("\n" + "="*80)
        print("STEP 4: LOAD TO DATABASE")
        print("="*80)
        
        # Try to load from embed checkpoint first, fallback to prepare
        checkpoint_data = self.checkpoint_manager.load_checkpoint("embed")
        embeddings_generated = True
        
        if checkpoint_data is None:
            print("ℹ️  No embed checkpoint found, trying prepare checkpoint...")
            checkpoint_data = self.checkpoint_manager.load_checkpoint("prepare")
            embeddings_generated = False
            
            if checkpoint_data is None:
                print("❌ No prepare or embed checkpoint found. Run 'prepare' or 'embed' step first.")
                return {"success": False, "error": "No checkpoint found"}
        
        df, metadata = checkpoint_data
        checkpoint_type = "embed" if embeddings_generated else "prepare"
        print(f"📂 Loaded {len(df)} records from {checkpoint_type} checkpoint")
        
        # Connect to database
        conn = self.get_db_connection()
        
        try:
            # Database operations
            if drop_table_first:
                self.drop_table(conn)
            
            if create_table:
                self.create_table(conn)
            
            # Load data
            loaded_count = self.load_data(conn, df, embeddings_generated)
            
            results = {
                "success": True,
                "loaded": loaded_count,
                "embeddings_included": embeddings_generated
            }
            
            print(f"\n✅ Load step completed: {loaded_count} records loaded")
            print("="*80 + "\n")
            
            return results
            
        finally:
            self.close_connection()
    
    def resume_pipeline(
        self,
        generate_embeddings: bool = True,
        create_table: bool = False,
        drop_table_first: bool = False,
        skip_existing: bool = True
    ) -> Dict[str, Any]:
        """
        Resume pipeline from the last checkpoint
        
        Automatically detects which steps have been completed and runs remaining steps
        """
        print("\n" + "="*80)
        print("RESUME PIPELINE FROM CHECKPOINTS")
        print("="*80)
        
        checkpoints = self.checkpoint_manager.list_checkpoints()
        
        # Determine which step to resume from
        if "embed" in checkpoints:
            print("✅ Found embed checkpoint - resuming from load step")
            return self.run_step("load", create_table=create_table, drop_table_first=drop_table_first)
        elif "prepare" in checkpoints:
            if generate_embeddings:
                print("✅ Found prepare checkpoint - resuming from embed step")
                embed_results = self.run_step("embed", save_checkpoint=True)
                if embed_results.get("success"):
                    return self.run_step("load", create_table=create_table, drop_table_first=drop_table_first)
                return embed_results
            else:
                print("✅ Found prepare checkpoint - resuming from load step")
                return self.run_step("load", create_table=create_table, drop_table_first=drop_table_first)
        elif "fetch" in checkpoints:
            print("✅ Found fetch checkpoint - resuming from prepare step")
            prepare_results = self.run_step("prepare", skip_existing=skip_existing, save_checkpoint=True)
            if not prepare_results.get("success"):
                return prepare_results
            
            if generate_embeddings:
                embed_results = self.run_step("embed", save_checkpoint=True)
                if not embed_results.get("success"):
                    return embed_results
            
            return self.run_step("load", create_table=create_table, drop_table_first=drop_table_first)
        else:
            print("❌ No checkpoints found. Please run 'fetch' step first.")
            return {"success": False, "error": "No checkpoints found"}
    


def load_config_from_env() -> RenderDatabaseConfig:
    """
    Load Render database configuration from environment variables
    
    Supports two methods:
    1. Database URL (recommended):
       - RENDER_DATABASE_URL or DATABASE_URL
    
    2. Individual credentials (fallback):
       - RENDER_DB_HOST
       - RENDER_DB_PORT (optional, default 5432)
       - RENDER_DB_NAME
       - RENDER_DB_USER
       - RENDER_DB_PASSWORD
    
    Returns:
        RenderDatabaseConfig: Database configuration
        
    Raises:
        ValueError: If no valid configuration is found
    """
    # Try to get database URL first (recommended approach)
    database_url = os.environ.get("RENDER_DATABASE_URL") or os.environ.get("DATABASE_URL")
    
    if database_url:
        return RenderDatabaseConfig(database_url=database_url)
    
    # Fallback to individual credentials
    host = os.environ.get("RENDER_DB_HOST")
    port = int(os.environ.get("RENDER_DB_PORT", 5432))
    dbname = os.environ.get("RENDER_DB_NAME")
    user = os.environ.get("RENDER_DB_USER")
    password = os.environ.get("RENDER_DB_PASSWORD")
    
    if not all([host, dbname, user, password]):
        raise ValueError(
            "Missing database configuration. Please provide either:\n"
            "1. RENDER_DATABASE_URL or DATABASE_URL environment variable, OR\n"
            "2. All of: RENDER_DB_HOST, RENDER_DB_NAME, RENDER_DB_USER, RENDER_DB_PASSWORD"
        )
    
    return RenderDatabaseConfig.from_components(
        host=host,
        user=user,
        password=password,
        dbname=dbname,
        port=port
    )


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Clinical Trials Data Pipeline - Fetch, process, and load clinical trials data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python load_to_render_postgres.py --step all
  
  # Run individual steps
  python load_to_render_postgres.py --step fetch --num-records 10000
  python load_to_render_postgres.py --step prepare
  python load_to_render_postgres.py --step embed
  python load_to_render_postgres.py --step load --create-table
  
  # Resume from last checkpoint
  python load_to_render_postgres.py --resume
  
  # List available checkpoints
  python load_to_render_postgres.py --list-checkpoints
  
  # Cleanup old checkpoints
  python load_to_render_postgres.py --cleanup-checkpoints
        """
    )
    
    parser.add_argument(
        "--step",
        choices=["fetch", "prepare", "embed", "load", "all"],
        default="all",
        help="Pipeline step to run (default: all)"
    )
    
    parser.add_argument(
        "--num-records",
        type=int,
        default=500000,
        help="Number of records to fetch (default: 500000)"
    )
    
    parser.add_argument(
        "--filter-by-start-date",
        action="store_true",
        default=True,
        help="Filter studies by start date (default: True)"
    )
    
    parser.add_argument(
        "--no-filter-by-start-date",
        dest="filter_by_start_date",
        action="store_false",
        help="Don't filter by start date"
    )
    
    parser.add_argument(
        "--start-year",
        type=int,
        default=2014,
        help="Filter studies starting after this year (default: 2014)"
    )
    
    parser.add_argument(
        "--generate-embeddings",
        action="store_true",
        default=True,
        help="Generate embeddings (default: True)"
    )
    
    parser.add_argument(
        "--no-embeddings",
        dest="generate_embeddings",
        action="store_false",
        help="Skip embedding generation"
    )
    
    parser.add_argument(
        "--create-table",
        action="store_true",
        default=False,
        help="Create database table if it doesn't exist"
    )
    
    parser.add_argument(
        "--drop-table",
        dest="drop_table_first",
        action="store_true",
        default=False,
        help="Drop existing table before creating new one"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip records that already exist in database (default: True)"
    )
    
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Process all records, even if they exist"
    )
    
    parser.add_argument(
        "--no-checkpoint",
        dest="save_checkpoint",
        action="store_false",
        default=True,
        help="Don't save checkpoint after step"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoint files (default: checkpoints)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume pipeline from last checkpoint"
    )
    
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List available checkpoints and exit"
    )
    
    parser.add_argument(
        "--cleanup-checkpoints",
        action="store_true",
        help="Remove old checkpoint files (keeps last 3 for each step)"
    )
    
    return parser.parse_args()


def list_checkpoints_cmd(checkpoint_dir: str):
    """List all available checkpoints"""
    manager = CheckpointManager(checkpoint_dir)
    checkpoints = manager.list_checkpoints()
    
    if not checkpoints:
        print("📂 No checkpoints found")
        return
    
    print("\n" + "="*80)
    print("AVAILABLE CHECKPOINTS")
    print("="*80)
    
    for step in ["fetch", "prepare", "embed"]:
        if step in checkpoints:
            print(f"\n📌 {step.upper()} Step:")
            for checkpoint in checkpoints[step]:
                print(f"  • {checkpoint.timestamp}: {checkpoint.num_records} records")
                print(f"    File: {checkpoint.file_path}")
    
    print("="*80 + "\n")


def cleanup_checkpoints_cmd(checkpoint_dir: str):
    """Cleanup old checkpoints"""
    manager = CheckpointManager(checkpoint_dir)
    print("\n🗑️  Cleaning up old checkpoints (keeping last 3 for each step)...\n")
    manager.cleanup_old_checkpoints(keep_last_n=3)
    print("\n✅ Cleanup complete\n")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Handle utility commands
    if args.list_checkpoints:
        list_checkpoints_cmd(args.checkpoint_dir)
        sys.exit(0)
    
    if args.cleanup_checkpoints:
        cleanup_checkpoints_cmd(args.checkpoint_dir)
        sys.exit(0)
    
    # Set up logging
    logger, log_file = setup_logging()
    logger.info("Starting clinical trials data pipeline for Render PostgreSQL")
    
    print(f"📝 Logging to: {log_file}\n")
    
    # Print configuration
    print("="*80)
    print("PIPELINE CONFIGURATION")
    print("="*80)
    print(f"  Step:                 {args.step}")
    if args.step in ["fetch", "all"]:
        print(f"  Records to fetch:     {args.num_records}")
        print(f"  Filter by start date: {args.filter_by_start_date}")
        if args.filter_by_start_date:
            print(f"  Start year filter:    > {args.start_year}")
    if args.step in ["embed", "all"]:
        print(f"  Generate embeddings:  {args.generate_embeddings}")
    if args.step in ["prepare", "all"]:
        print(f"  Skip existing:        {args.skip_existing}")
    if args.step in ["load", "all"]:
        print(f"  Drop existing table:  {args.drop_table_first}")
        print(f"  Create table:         {args.create_table}")
    print(f"  Checkpoint directory: {args.checkpoint_dir}")
    print(f"  Save checkpoints:     {args.save_checkpoint}")
    print("="*80 + "\n")
    
    try:
        # Determine if database config is needed for this step
        # Only fetch and embed steps don't need database access
        needs_db = args.step not in ["fetch", "embed"] or args.resume
        if args.step == "prepare" and not args.skip_existing:
            needs_db = False
        
        # Load database configuration only if needed
        if needs_db:
            db_config = load_config_from_env()
        else:
            # Create a dummy config for steps that don't need database
            db_config = RenderDatabaseConfig(database_url="postgresql://dummy:dummy@localhost:5432/dummy")
        
        # Initialize pipeline
        pipeline = ClinicalTrialsDataPipeline(db_config, logger, args.checkpoint_dir)
        
        # Run pipeline based on mode
        if args.resume:
            print("🔄 Resuming from last checkpoint...\n")
            results = pipeline.resume_pipeline(
                generate_embeddings=args.generate_embeddings,
                create_table=args.create_table,
                drop_table_first=args.drop_table_first,
                skip_existing=args.skip_existing
            )
        else:
            results = pipeline.run_step(
                step=args.step,
                num_records=args.num_records,
                generate_embeddings=args.generate_embeddings,
                create_table=args.create_table,
                drop_table_first=args.drop_table_first,
                filter_by_start_date=args.filter_by_start_date,
                start_year=args.start_year,
                skip_existing=args.skip_existing,
                save_checkpoint=args.save_checkpoint
            )
        
        logger.info(f"Pipeline completed. Results: {results}")
        
        if results.get("success", True):
            print("✅ Pipeline execution completed successfully")
            sys.exit(0)
        else:
            print("⚠️  Pipeline execution completed with warnings")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
