import pandas as pd
import requests
import json
import os
import psycopg
from psycopg import sql
import streamlit as st
import numpy as np
# import matplotlib.pyplot as plt
import datetime
import math


def fetch_clinical_trials(total_limit=1):
    """
    Fetch clinical trials from the ClinicalTrials.gov API with pagination support

    Parameters:
    total_limit (int): Total number of records to fetch across all pages

    Returns:
    pandas.DataFrame: DataFrame containing the clinical trials data with selected columns
    """
    # API endpoint for ClinicalTrials.gov
    base_url = "https://clinicaltrials.gov/api/v2/studies"

    # Parameters for the initial request
    params = {
        "format": "json",
        "pageSize": min(total_limit, 2000),  # API typically limits page size
    }

    all_studies = []
    records_fetched = 0

    # Continue fetching until we reach the total limit or there are no more pages
    while records_fetched < total_limit:
        # Make the request
        response = requests.get(base_url, params=params)

        # Check if the request was successful
        if response.status_code != 200:
            print(f"Failed to fetch data: {response.status_code}")
            break

        # Parse the JSON response
        data = response.json()

        # Extract the studies
        studies = data.get("studies", [])

        # Add studies to our collection
        all_studies.extend(studies)
        records_fetched += len(studies)

        print(f"Fetched {len(studies)} records, total: {records_fetched}")

        # Check if there's a next page
        next_page_token = data.get("nextPageToken")
        if not next_page_token or records_fetched >= total_limit:
            break

        # Update parameters for the next request
        params["pageToken"] = next_page_token
        params["pageSize"] = min(total_limit - records_fetched, 2000)

    # Convert to DataFrame
    if all_studies:
        # Create DataFrame with all data first
        df = pd.json_normalize(all_studies)
        # Define the columns we want to keep
        columns_to_keep = [
            # identificationModule
            "protocolSection.identificationModule.nctId",  # Adding NCT ID as it's typically needed as a unique identifier
            "protocolSection.identificationModule.briefTitle",
            "protocolSection.identificationModule.officialTitle",
            # statusModule
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
            # sponsorCollaboratorsModule
            "protocolSection.sponsorCollaboratorsModule.leadSponsor.name",
            "protocolSection.sponsorCollaboratorsModule.leadSponsor.class",
            "protocolSection.sponsorCollaboratorsModule.collaborators",
            # interventionsModule
            "protocolSection.armsInterventionsModule.interventions",
            # descriptionModule
            "protocolSection.descriptionModule.briefSummary",
            "protocolSection.descriptionModule.detailedDescription",
            # conditionsModule
            "protocolSection.conditionsModule.conditions",
            # designModule
            "protocolSection.designModule.studyType",
            "protocolSection.designModule.phases",
            "protocolSection.designModule.designInfo.allocation",
            "protocolSection.designModule.enrollmentInfo.count",
            # eligibilityModule
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
            # referencesModule
            "protocolSection.referencesModule.references",
            "protocolSection.referencesModule.seeAlsoLinks",
            "protocolSection.referencesModule.availIpds",
        ]

        print(f"Need to keep {len(columns_to_keep)} columns from the fetched data.")

        # Filter columns (only keep those that exist in the dataframe)
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        filtered_df = df[existing_columns]

        print(f"Filtered DataFrame to {len(existing_columns)} columns.")

        print(
            f"Successfully fetched a total of {len(filtered_df)} clinical trials with {len(existing_columns)} selected columns."
        )
        return filtered_df
    else:
        print("No studies were fetched.")
        return None


def create_clinical_trials_table(conn):
    """
    Create the clinical_trials table in PostgreSQL if it doesn't exist
    
    Parameters:
    conn: PostgreSQL connection object
    """
    with conn.cursor() as cur:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS clinical_trials (
            id SERIAL PRIMARY KEY,
            nct_id TEXT UNIQUE,
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
            brief_summary TEXT,
            detailed_description TEXT,
            conditions JSONB,
            study_type TEXT,
            healthy_volunteers BOOLEAN,
            phases JSONB,
            allocation TEXT,
            enrollment_count FLOAT,
            eligibility_criteria TEXT,
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
            drug_description TEXT
        )
        """)
        
        conn.commit()
        print("Created clinical_trials table if it didn't exist")


def drop_clinical_trials_table(conn):
    """Drop the clinical_trials table if it exists."""
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS clinical_trials")
        conn.commit()
        print("Dropped clinical_trials table if it existed.")


def load_clinical_trials_to_db(df, conn):
    """
    Load clinical trials data from DataFrame to PostgreSQL using psycopg3
    
    Parameters:
    df: DataFrame containing clinical trials data
    conn: PostgreSQL connection object
    
    Returns:
    int: Number of records inserted
    """
    # Filter out records that don't have drug interventions
    def has_drug_interventions(interventions):
        # Handle None first
        if interventions is None:
            return False
        
        # Handle pandas NA values - check if it's a scalar NA
        try:
            if pd.isna(interventions):
                return False
        except (TypeError, ValueError):
            # If pd.isna raises an error, the value is likely not a scalar
            pass
        
        # Handle string representations of JSON
        if isinstance(interventions, str):
            if not interventions.strip():  # Empty string
                return False
            try:
                interventions = json.loads(interventions)
            except (json.JSONDecodeError, TypeError):
                return False
        
        # Handle non-list types
        if not isinstance(interventions, list):
            return False
        
        # Check if any intervention is a drug
        return any(
            isinstance(intervention, dict) and intervention.get('type') == 'DRUG' 
            for intervention in interventions
        )
    
    # Check if the interventions column exists
    if 'protocolSection.armsInterventionsModule.interventions' not in df.columns:
        print("No interventions column found in DataFrame")
        return 0
    
    # Filter DataFrame to only include records with drug interventions
    df_filtered = df[df['protocolSection.armsInterventionsModule.interventions'].apply(has_drug_interventions)].copy()
    
    print(f"Filtered from {len(df)} to {len(df_filtered)} records with drug interventions")
    
    # Filter to only keep records with start_date after 2015
    def start_date_after(date_str, limit_year=2015):
        if pd.isna(date_str) or not date_str:
            return False
        try:
            # Accept YYYY-MM-DD, YYYY-MM, or YYYY
            if isinstance(date_str, datetime.date):
                year = date_str.year
            elif len(date_str) == 10:
                year = int(date_str[:4])
            elif len(date_str) == 7:
                year = int(date_str[:4])
            elif len(date_str) == 4:
                year = int(date_str)
            else:
                return False
            return year > limit_year
        except Exception:
            return False

    if 'protocolSection.statusModule.startDateStruct.date' in df_filtered.columns:
        df_filtered = df_filtered[df_filtered['protocolSection.statusModule.startDateStruct.date'].apply(start_date_after)]
        print(f"Filtered to {len(df_filtered)} records with start_date after 2015")
    else:
        print("start_date column not found for filtering by year")

    if len(df_filtered) == 0:
        print("No records with drug interventions found")
        return 0
    
    # Extract drug information
    def extract_drug_info(interventions):
        # Handle None first
        if interventions is None:
            return None, None
        
        # Handle pandas NA values
        try:
            if pd.isna(interventions):
                return None, None
        except (TypeError, ValueError):
            pass
        
        # Handle string representations of JSON
        if isinstance(interventions, str):
            if not interventions.strip():
                return None, None
            try:
                interventions = json.loads(interventions)
            except (json.JSONDecodeError, TypeError):
                return None, None
        
        # Handle non-list types
        if not isinstance(interventions, list):
            return None, None
        
        drug_names = []
        drug_descriptions = []
        
        for intervention in interventions:
            if isinstance(intervention, dict) and intervention.get('type') == 'DRUG':
                name = intervention.get('name')
                description = intervention.get('description')
                if name:
                    drug_names.append(name)
                if description:
                    drug_descriptions.append(description)
        
        return ('; '.join(drug_names) if drug_names else None,
                '; '.join(drug_descriptions) if drug_descriptions else None)
    
    # Apply drug extraction
    drug_info = df_filtered['protocolSection.armsInterventionsModule.interventions'].apply(extract_drug_info)
    df_filtered['drug_name'] = [info[0] for info in drug_info]
    df_filtered['drug_description'] = [info[1] for info in drug_info]
    
    # Create a mapping from DataFrame columns to database columns
    column_mapping = {
        "protocolSection.identificationModule.nctId": "nct_id",
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
        "drug_name": "drug_name",
        "drug_description": "drug_description"
    }
    
    # Rename columns in DataFrame to match database columns
    renamed_df = df_filtered.copy()
    for df_col, db_col in column_mapping.items():
        if df_col in renamed_df.columns:
            renamed_df[db_col] = renamed_df[df_col]
    
    # Define JSONB columns that need special handling (removed interventions)
    jsonb_columns = ["collaborators", "conditions", "phases", "std_ages", "study_references", "see_also_links", "avail_ipds"]

    # Print JSONB columns before conversion for debugging
    print("JSONB columns before conversion:")
    for col in jsonb_columns:
        if col in renamed_df.columns:
            print(f"\nColumn: {col}")
            print(renamed_df[col].head())

    # Convert complex data types to proper format
    def safe_json_serialize(obj):
        """Safely serialize objects to JSON-compatible format"""
        # Handle None first
        if obj is None:
            return None

        # Handle numpy arrays before pd.isna check
        if isinstance(obj, np.ndarray):
            return [safe_json_serialize(item) for item in obj.tolist()]

        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return [safe_json_serialize(item) for item in obj]

        # Handle dictionaries
        if isinstance(obj, dict):
            return {k: safe_json_serialize(v) for k, v in obj.items()}

        # Now safe to check pd.isna for scalar values
        if pd.isna(obj):
            return None

        # Handle numpy scalar types
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

    # Process JSONB columns with better handling
    for col in jsonb_columns:
        if col in renamed_df.columns:
            def process_json_column(x):
                if x is None:
                    return None
                # Handle the case where x might be a numpy array or complex object
                try:
                    serialized = safe_json_serialize(x)
                    return json.dumps(serialized) if serialized is not None else None
                except Exception as e:
                    # Fallback: convert to string representation
                    return json.dumps(str(x))

            renamed_df[col] = renamed_df[col].apply(process_json_column)

    # Print JSONB columns after conversion for debugging
    print("JSONB columns after conversion:")
    for col in jsonb_columns:
        if col in renamed_df.columns:
            print(f"\nColumn: {col}")
            print(renamed_df[col].head())
    
    # Get only the columns that match the database schema
    db_columns = [
        "nct_id", "brief_title", "official_title", "overall_status",
        "start_date", "start_date_type", "primary_completion_date", "primary_completion_date_type",
        "completion_date", "completion_date_type", "study_first_submit_date",
        "last_update_date", "last_update_date_type", "lead_sponsor_name",
        "lead_sponsor_class", "collaborators", "brief_summary", "detailed_description",
        "conditions", "study_type", "phases", "allocation", "enrollment_count",
        "eligibility_criteria", "healthy_volunteers", 
        "gender_based", "gender_description",
        "sex", "minimum_age", "maximum_age", "std_ages", "study_population",
        "sampling_method", "study_references", "see_also_links", "avail_ipds", 
        "drug_name", "drug_description"
    ]
    
    # Filter columns to only include those in renamed_df
    cols_to_use = [col for col in db_columns if col in renamed_df.columns]
    final_df = renamed_df[cols_to_use].copy()
    
    # Convert all remaining numpy types to Python native types
    def convert_value(value):
        """Convert numpy types and other problematic types to Python native types"""
        # print(f"Converting value: {value} (type: {type(value)})")
        if value is None:
            return None
        # Explicitly check for float NaN
        elif isinstance(value, float) and math.isnan(value):
            return None
        elif pd.isna(value):
            return None
        # Convert numpy integer to Python int
        elif isinstance(value, (np.integer, np.int64)):
            return int(value)
        # Convert numpy float to Python float or int if it's a whole number
        elif isinstance(value, (np.floating, np.float64)):
            if value.is_integer():
                return int(value)
            return float(value)
        # Convert numpy bool to Python bool
        elif isinstance(value, np.bool_):
            return bool(value)
        # Convert numpy str to Python str
        elif isinstance(value, np.str_):
            return str(value)
        # Return native Python types as-is or convert float to int if appropriate
        elif isinstance(value, float):
            if value.is_integer():
                return int(value)
            return value
        elif isinstance(value, (int, bool, str)):
            return value
        # For anything else, return None
        else:
            return None
    
    # Apply conversion to all columns
    # Print before conversion for enrollment_count
    if "enrollment_count" in final_df.columns:
        # print("enrollment_count before conversion:")
        # print(final_df["enrollment_count"].head())
    
        # Convert the enrollment_count column explicitly
        final_df["enrollment_count"] = final_df["enrollment_count"].apply(lambda x: 
            None if pd.isna(x) else 
            int(x) if isinstance(x, (float, np.floating)) and not math.isnan(x) else 
            convert_value(x)
        )
        
        # Force the column to be integer type with None for missing values
        final_df["enrollment_count"] = pd.to_numeric(final_df["enrollment_count"], errors='coerce').astype('Int64')
        
        # print("enrollment_count after conversion:")
        # print(final_df["enrollment_count"].head())
    
    # Convert all columns to native Python types
    for col in final_df.columns:
        final_df[col] = final_df[col].apply(convert_value)
    
    def parse_date(date_str):
        """Convert YYYY-MM-DD or YYYY-MM or YYYY to datetime.date, or return None if invalid."""
        if pd.isna(date_str) or not date_str:
            return None
        try:
            # If already a date object, return as is
            if isinstance(date_str, datetime.date):
                return date_str
            # If format is YYYY-MM-DD
            if len(date_str) == 10:
                return datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            # If format is YYYY-MM
            elif len(date_str) == 7:
                return datetime.datetime.strptime(date_str + "-01", "%Y-%m-%d").date()
            # If format is YYYY
            elif len(date_str) == 4:
                return datetime.datetime.strptime(date_str + "-01-01", "%Y-%m-%d").date()
        except Exception:
            return None
        return None

    # Convert date columns to proper date format
    date_columns = [
        "start_date",
        "primary_completion_date",
        "completion_date",
        "study_first_submit_date",
        "last_update_date"
    ]
    for col in date_columns:
        if col in final_df.columns:
            final_df[col] = final_df[col].apply(parse_date)
    
    def parse_age(age_str):
        """
        Convert age strings like '18 Years', '1 Year', 'N/A', etc. to integer values in years.
        Returns None for invalid or missing values.
        """
        if pd.isna(age_str) or not isinstance(age_str, str):
            return None
            
        # Remove any trailing periods and convert to lowercase
        age_str = age_str.rstrip('.').lower()
        
        # Common patterns to handle
        if age_str in ('n/a', '', 'na'):
            return None
            
        try:
            # Extract numeric value
            number = ''.join(c for c in age_str if c.isdigit() or c == '.')
            if not number:
                return None
                
            value = float(number)
            
            # Convert to years based on unit
            if 'month' in age_str:
                value = int(value / 12)
            elif 'week' in age_str:
                value = int(value / 52)
            elif 'day' in age_str:
                value = int(value / 365)
            else:  # Assume years
                value = int(value)
                
            return value
        except:
            return None

    # Convert age columns to integers
    age_columns = ["minimum_age", "maximum_age"]
    for col in age_columns:
        if col in final_df.columns:
            final_df[col] = final_df[col].apply(parse_age)
            
            # Print conversion results for debugging
            print(f"{col} after conversion:")
            print(final_df[col].head())
    
    # Convert DataFrame to list of dictionaries
    records = final_df.to_dict('records')
    
    # Insert data into the database
    with conn.cursor() as cur:
        columns_str = ', '.join(cols_to_use)
        placeholders = ', '.join(['%s'] * len(cols_to_use))
        
        # Upsert query with ON CONFLICT DO UPDATE
        upsert_query = f"""
        INSERT INTO clinical_trials ({columns_str})
        VALUES ({placeholders})
        ON CONFLICT (nct_id) 
        DO UPDATE SET {', '.join([f"{col} = EXCLUDED.{col}" for col in cols_to_use if col != 'nct_id'])}
        """
        
        # Execute batch insert
        for record in records:
            values = [record.get(col) for col in cols_to_use]
            try:
                cur.execute(upsert_query, values)
            except Exception as e:
                print("Error inserting record:", record)
                print("Values:", values)
                print("Exception:", e)
                raise
        
        conn.commit()
        
    inserted_rows = len(records)
    print(f"Inserted or updated {inserted_rows} records in the clinical_trials table")
    return inserted_rows


def get_db_connection():
    """
    Get a connection to the PostgreSQL database using environment variables or Streamlit secrets
    
    Returns:
    connection: PostgreSQL connection object
    """
    # Try to get connection parameters from Streamlit secrets first
    try:
        host = st.secrets.get("POSTGRES_HOST", "localhost")
        port = st.secrets.get("POSTGRES_PORT", 5432)
        dbname = st.secrets.get("POSTGRES_DBNAME")
        user = st.secrets.get("POSTGRES_USER")
        password = st.secrets.get("POSTGRES_PASSWORD")
    except Exception:
        # If not using Streamlit or secrets not available, try environment variables
        host = os.environ.get("POSTGRES_HOST", "localhost")
        port = os.environ.get("POSTGRES_PORT", 5432)
        dbname = os.environ.get("POSTGRES_DBNAME")
        user = os.environ.get("POSTGRES_USER")
        password = os.environ.get("POSTGRES_PASSWORD")
    
    # Connect to the database using psycopg3
    conn = psycopg.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )
    
    return conn


if __name__ == "__main__":
    # Fetch clinical trials
    print("Fetching clinical trials data...")
    df = fetch_clinical_trials(200000)

    if df is not None:
        print("\nDataFrame fetched with selected columns")
        print("\nColumns in the DataFrame:")
        print(df.columns.tolist())
        # df["protocolSection.referencesModule.availIpds"]
        # Plot histogram of enrollment_count
        # if "protocolSection.designModule.enrollmentInfo.count" in df.columns:
        #     enrollment_counts = pd.to_numeric(df["protocolSection.designModule.enrollmentInfo.count"], errors="coerce")
        #     plt.figure(figsize=(10, 6))
        #     plt.hist(enrollment_counts.dropna(), bins=50, color='skyblue', edgecolor='black')
        #     plt.title("Histogram of Enrollment Count")
        #     plt.xlabel("Enrollment Count")
        #     plt.ylabel("Frequency")
        #     plt.savefig("enrollment_count_hist.png")  # Save plot to file
        #     print("Histogram saved as enrollment_count_hist.png")
        # else:
        #     print("enrollment_count column not found in DataFrame.")

        # Connect to database
        try:
            print("Connecting to PostgreSQL database...")
            conn = get_db_connection()
            
            # Drop table if it exists
            print("Dropping table if it exists...")
            drop_clinical_trials_table(conn)
            
            # Create table if it doesn't exist
            print("Creating table...")
            create_clinical_trials_table(conn)
    
            # Load data to database
            print("Loading data to database...")
            inserted_rows = load_clinical_trials_to_db(df, conn)
            print(f"Successfully loaded {inserted_rows} clinical trials to PostgreSQL")
            
            # Close connection
            conn.close()
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("No data to load to database")
