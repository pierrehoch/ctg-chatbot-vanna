import streamlit as st
import os
# from vanna.rag import VannaRAG
from vanna.chromadb import ChromaDB_VectorStore
from vanna.openai import OpenAI_Chat
from datetime import date as dt_date, timedelta
import pandas as pd
import numpy as np
from src.utils.get_embedding import get_openai_embedding


class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        # Initialize ChromaDB_VectorStore with a specific path for the database
        # You can change 'chroma_db' to any path you prefer
        os.makedirs('chroma_db', exist_ok=True)

        ChromaDB_VectorStore.__init__(
            self, 
            config={
                'path': 'chroma_db',
                'allow_reset': False  # Prevent resetting the database
            }
        )

        OpenAI_Chat.__init__(
            self, 
            config={
                'api_key': st.secrets.get("OPENAI_API_KEY"),
                'model': 'gpt-4o-mini',  # Use the smaller model
                'chunk_size': 500,  # Use smaller chunks
                'chunk_overlap': 50  # Use less overlap
            }
        )

    def is_trained(self):
        """Check if the model has already been trained by checking if collections exist"""
        try:
            collections = self._chromadb_client.list_collections()
            return len(collections) > 0
        except:
            return False

def preprocess_dataframe(df, max_rows=10, max_cols=10, max_text_length=500):
    """
    Preprocesses a dataframe to reduce its size before sending to API.
    
    Args:
        df (pandas.DataFrame): The dataframe to preprocess
        max_rows (int): Maximum number of rows to include
        max_cols (int): Maximum number of columns to include
        max_text_length (int): Maximum length for text fields
        
    Returns:
        pandas.DataFrame: The preprocessed dataframe
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Limit rows
    sample_df = df.head(max_rows) if len(df) > max_rows else df.copy()
    
    # Limit columns if needed
    if len(sample_df.columns) > max_cols:
        sample_df = sample_df.iloc[:, :max_cols]
    
    # Process each column to reduce size
    for col in sample_df.columns:
        # Handle text columns - truncate long strings
        if sample_df[col].dtype == 'object':
            sample_df[col] = sample_df[col].apply(
                lambda x: str(x)[:max_text_length] + '...' if isinstance(x, str) and len(str(x)) > max_text_length else x
            )
        
        # Handle JSON/dict columns - convert to string representation
        elif any(isinstance(val, (dict, list)) for val in sample_df[col].dropna()):
            sample_df[col] = sample_df[col].apply(
                lambda x: str(x)[:max_text_length] + '...' if x is not None and len(str(x)) > max_text_length else x
            )
        
        # Handle datetime columns - keep as is
        elif pd.api.types.is_datetime64_any_dtype(sample_df[col]):
            continue
        
        # Handle numeric columns - keep as is
        elif pd.api.types.is_numeric_dtype(sample_df[col]):
            continue
            
        # For any other types, convert to string and truncate if needed
        else:
            sample_df[col] = sample_df[col].apply(
                lambda x: str(x)[:max_text_length] + '...' if x is not None and len(str(x)) > max_text_length else x
            )
    
    return sample_df

@st.cache_resource(ttl=3600)
def setup_vanna():
    vn = MyVanna()

    # Only train if the database doesn't already have content
    if not vn.is_trained():
        # st.write("Training model with schema and documentation...")
        
        db_host = st.secrets.get("POSTGRES_HOST", "localhost")
        db_port = st.secrets.get("POSTGRES_PORT", 5432)
        db_name = st.secrets.get("POSTGRES_DBNAME")
        db_user = st.secrets.get("POSTGRES_USER")
        db_password = st.secrets.get("POSTGRES_PASSWORD")
        db_sslmode = st.secrets.get("POSTGRES_SSLMODE", "prefer")  # Add this line

        # Add this debugging code right before the connection attempt
        print(f"Host: {db_host}")
        print(f"Port: {db_port}")
        print(f"DB Name: {db_name}")
        print(f"User: {db_user}")
        print(f"Password length: {len(db_password) if db_password else 0}")
        
        # Check for encoding issues in password
        if db_password:
            try:
                # Try to encode/decode the password to check for issues
                encoded_pass = db_password.encode('utf-8')
                decoded_pass = encoded_pass.decode('utf-8')
                print(f"Password encoding check: OK")
            except UnicodeError as e:
                print(f"Password encoding issue: {e}")

        # Connect with SSL mode parameter and explicit encoding
        try:
            # First, let's try to bypass vanna's connection method entirely
            import psycopg2
            
            # Create a minimal connection string with explicit encoding
            conn_params = {
                'host': db_host,
                'port': db_port,
                'dbname': db_name,
                'user': db_user,
                'password': db_password,
                'sslmode': db_sslmode,
                'client_encoding': 'utf8'
            }
            
            # print(f"Trying direct psycopg2 connection...")
            
            # # Test connection with psycopg2 directly
            # test_conn = psycopg2.connect(**conn_params)
            # test_conn.close()
            # print("Direct psycopg2 connection successful!")
            
            # Now try vanna's connection without sslmode first
            print("Trying vanna connection without sslmode...")
            vn.connect_to_postgres(
                host=db_host, 
                port=db_port, 
                dbname=db_name, 
                user=db_user, 
                password=db_password
            )
            
        except UnicodeDecodeError as e:
            print(f"Unicode error details:")
            print(f"Position: {e.start}")
            print(f"Reason: {e.reason}")
            
            # Try with ASCII encoding fallback
            try:
                print("Trying with ASCII fallback...")
                ascii_password = db_password.encode('ascii', errors='ignore').decode('ascii')
                vn.connect_to_postgres(
                    host=db_host, 
                    port=db_port, 
                    dbname=db_name, 
                    user=db_user, 
                    password=ascii_password
                )
            except Exception as ascii_e:
                print(f"ASCII fallback failed: {ascii_e}")
                
                # Final attempt: use environment variables
                print("Final attempt with environment variables...")
                import os
                os.environ['PGHOST'] = db_host
                os.environ['PGPORT'] = str(db_port)
                os.environ['PGDATABASE'] = db_name
                os.environ['PGUSER'] = db_user
                os.environ['PGPASSWORD'] = db_password
                os.environ['PGSSLMODE'] = db_sslmode
                
                try:
                    # Try connecting without parameters (using env vars)
                    test_conn = psycopg2.connect()
                    test_conn.close()
                    
                    # If successful, manually set vanna's connection
                    vn._conn = psycopg2.connect()
                    print("Connection successful using environment variables!")
                    
                except Exception as env_e:
                    print(f"Environment variable approach failed: {env_e}")
                    raise RuntimeError("All connection attempts failed due to Unicode encoding issues")
                    
        except Exception as e:
            print(f"Other connection error: {e}")
            raise
        vn.train(ddl="""
        CREATE TABLE clinical_trials (
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
            drug_description TEXT
        """)
        
      
        # Time-based queries and best practices
        vn.train(documentation="""
        TIME-BASED QUERY EXAMPLES AND BEST PRACTICES:
        ---------------------------------------------
        
        - To find the soonest upcoming trials (trials starting after today, ordered by start date):
        
          SQL: 
          SELECT nct_id, brief_title, conditions, drug_name, phases, start_date, completion_date, overall_status, enrollment_count
          FROM clinical_trials 
          WHERE start_date >= '{today}' 
          ORDER BY start_date ASC
        
        - To find trials that are currently recruiting and will start soon:
        
          SQL:
          SELECT nct_id, brief_title, conditions, overall_status, start_date, completion_date, phases
          FROM clinical_trials 
          WHERE overall_status = 'RECRUITING' 
            AND start_date >= '{today}' 
          ORDER BY start_date ASC
        
        - To find trials that are about to complete soon:
        
          SQL:
          SELECT nct_id, brief_title, conditions, completion_date, overall_status, phases
          FROM clinical_trials 
          WHERE completion_date >= '{today}' 
          ORDER BY completion_date ASC
        
        - To find trials that started in the past month:
        
          SQL:
          SELECT nct_id, brief_title, conditions, start_date, overall_status, phases
          FROM clinical_trials 
          WHERE start_date >= '{one_month_ago}' 
            AND start_date < '{today}'
        
        - To find trials that are ongoing as of today:
        
          SQL:
          SELECT nct_id, brief_title, conditions, start_date, completion_date, overall_status, phases
          FROM clinical_trials 
          WHERE start_date <= '{today}' 
            AND (completion_date IS NULL OR completion_date >= '{today}')
        
        - Always use the date provided in the question (e.g., "as of 2025-06-23") as the reference for 'today'.
        - Use ORDER BY with ASC for soonest/upcoming, DESC for most recent/past.
        - Only use LIMIT if the user specifically asks for a limited number of results (e.g., "top 10", "first 5").
        
        Replace {today} with the current date (provided in the prompt).
        Replace {one_month_ago} with the date one month before today.
        
        EXAMPLES:
        ---------
        Question: "What are the soonest upcoming trials?"
        SQL: SELECT nct_id, brief_title, conditions, drug_name, phases, start_date, completion_date, overall_status
             FROM clinical_trials WHERE start_date >= '{today}' ORDER BY start_date ASC
        
        Question: "Which trials are about to complete?"
        SQL: SELECT nct_id, brief_title, conditions, completion_date, overall_status, phases
             FROM clinical_trials WHERE completion_date >= '{today}' ORDER BY completion_date ASC
        
        Question: "Show me the most recently started trials"
        SQL: SELECT nct_id, brief_title, conditions, start_date, overall_status, phases
             FROM clinical_trials WHERE start_date <= '{today}' ORDER BY start_date DESC
        
        Question: "Show me the top 10 upcoming trials"
        SQL: SELECT nct_id, brief_title, conditions, drug_name, phases, start_date, completion_date, overall_status
             FROM clinical_trials WHERE start_date >= '{today}' ORDER BY start_date ASC LIMIT 10
        """)
        

                # Update the phases documentation with proper JSONB querying examples
        vn.train(documentation="""
        JSONB Column Query Examples:
        ---------------------------
        
        For the 'phases' column (JSONB type):
        - To find trials in a specific phase: WHERE phases @> '["PHASE2"]'
        - To find trials in multiple phases: WHERE phases @> '["PHASE1"]' OR phases @> '["PHASE2"]'
        - To find trials containing any of several phases: WHERE phases ?| array['PHASE1', 'PHASE2']
        - To check if phases contains a value: WHERE phases ? 'PHASE2'
        
        For the 'conditions' column (JSONB type):
        - To find trials studying a condition: WHERE conditions @> '["Diabetes"]'
        - Case-insensitive condition search: WHERE LOWER(conditions::text) LIKE LOWER('%diabetes%')
        
        For the 'collaborators' column (JSONB type):
        - To find specific collaborator: WHERE collaborators @> '[{"name": "Hospital Name"}]'
        - To search collaborator names: WHERE collaborators::text ILIKE '%hospital%'
        
        For the 'std_ages' column (JSONB type):
        - To find trials for adults: WHERE std_ages @> '["ADULT"]'
        - To find trials for children: WHERE std_ages @> '["CHILD"]'
        
        IMPORTANT JSONB OPERATORS:
        - @> : contains (left contains right)
        - ? : key/element exists
        - ?| : any of the keys/elements exist
        - ?& : all of the keys/elements exist
        - :: : cast operator (use carefully with JSONB)
        
        NEVER use ::text[] casting on JSONB columns. Instead use JSONB operators like @>, ?, ?|, ?&
        """)
        
        # Add specific examples for common phase queries
        vn.train(documentation="""
        COMMON PHASE QUERY PATTERNS:
        ----------------------------
        
        Question: "Show me Phase 2 trials"
        SQL: SELECT nct_id, brief_title, phases, conditions, drug_name, start_date, completion_date, overall_status
             FROM clinical_trials WHERE phases @> '["PHASE2"]'
        
        Question: "Find Phase 1 or Phase 2 trials"  
        SQL: SELECT nct_id, brief_title, phases, conditions, drug_name, overall_status
             FROM clinical_trials WHERE phases @> '["PHASE1"]' OR phases @> '["PHASE2"]'
        
        Question: "Trials in early phases"
        SQL: SELECT nct_id, brief_title, phases, conditions, drug_name, start_date, overall_status
             FROM clinical_trials WHERE phases ?| array['PHASE1', 'PHASE2']
        
        Question: "Phase 3 trials for diabetes"
        SQL: SELECT nct_id, brief_title, phases, conditions, drug_name, start_date, completion_date
             FROM clinical_trials WHERE phases @> '["PHASE3"]' AND LOWER(conditions::text) LIKE '%diabetes%'
        
        Question: "What phases are available?"
        SQL: SELECT DISTINCT jsonb_array_elements_text(phases) as phase FROM clinical_trials WHERE phases IS NOT NULL
        """)

        # Train with documentation in smaller chunks
        # Chunk 1: Fields 1-10
        vn.train(documentation="""
        1. id
        Type: SERIAL PRIMARY KEY

        Description: A unique auto-incrementing identifier for each record in the database.

        Example: 1, 2, 3

        Usage: Used internally for database operations; not typically exposed to users.

        2. nct_id
        Type: TEXT UNIQUE

        Description: The unique identifier assigned by ClinicalTrials.gov to each clinical trial.

        Example: NCT12345678

        Usage: Used to reference and link to the trial on the ClinicalTrials.gov website.
        
        3. org_study_id
        Type: TEXT
        
        Description: The organization's internal study identifier.
        
        Example: "ABC-123-XYZ"
        
        Usage: Used to reference the study within the sponsoring organization.

        4. brief_title
        Type: TEXT

        Description: A short, descriptive title for the clinical trial.

        Example: "Efficacy of Drug X in Treating Condition Y"

        Usage: Provides a quick reference for the trial's purpose.

        5. official_title
        Type: TEXT

        Description: The formal, official title of the clinical trial.

        Example: "A Randomized, Double-Blind, Placebo-Controlled Study of Drug X in Patients with Condition Y"

        Usage: Used in official documentation and regulatory submissions.

        6. overall_status
        Type: TEXT

        Description: The current status of the clinical trial (e.g., "RECRUITING", "COMPLETED", "NOT_YET_RECRUITING", "UNKNOWN", "WITHDRAWN", "TERMINATED").

        Example: "RECRUITING"

        Usage: Indicates the trial's current phase or stage.

        7. start_date
        Type: DATE

        Description: The date when the trial started.

        Example: 2023-01-15

        Usage: Provides timeline information for the trial.

        8. start_date_type
        Type: TEXT

        Description: The type of start date (e.g., "Actual", "Anticipated").

        Example: "Actual"

        Usage: Clarifies whether the date is confirmed or estimated.

        9. primary_completion_date
        Type: DATE

        Description: The date when the primary outcome measure is assessed.

        Example: 2024-06-30

        Usage: Marks a key milestone in the trial timeline.

        10. primary_completion_date_type
        Type: TEXT

        Description: The type of primary completion date (e.g., "Actual", "Anticipated").

        Example: "Actual"

        Usage: Clarifies whether the date is confirmed or estimated.

        11. completion_date
        Type: DATE

        Description: The date when the trial is completed.

        Example: 2024-12-31

        Usage: Indicates the end of the trial.
            """)

        # Chunk 2: Fields 11-20
        vn.train(documentation="""
    12. completion_date_type
    Type: TEXT

    Description: The type of completion date (e.g., "Actual", "Anticipated").

    Example: "Anticipated"

    Usage: Clarifies whether the date is confirmed or estimated.

    13. study_first_submit_date
    Type: DATE

    Description: The date when the trial was first submitted to ClinicalTrials.gov.

    Example: 2022-11-10

    Usage: Tracks the submission timeline.

    14. last_update_date
    Type: DATE

    Description: The date when the trial record was last updated.

    Example: 2024-03-05

    Usage: Indicates the most recent update to the trial information.

    15. last_update_date_type
    Type: TEXT

    Description: The type of last update date (e.g., "Actual", "Anticipated").

    Example: "Actual"

    Usage: Clarifies whether the date is confirmed or estimated.

    16. lead_sponsor_name
    Type: TEXT

    Description: The name of the primary sponsor of the clinical trial.

    Example: "National Institutes of Health"

    Usage: Identifies the main organization responsible for the trial.

    17. lead_sponsor_class
    Type: TEXT

    Description: The classification of the lead sponsor (e.g., "NIH", "Industry", "Other").

    Example: "NIH"

    Usage: Categorizes the sponsor for analysis and reporting.

    18. collaborators 
    Type: JSONB

    Description: A JSON array or object listing organizations collaborating on the trial.

    Example: 
    [
        {
            "name": "Massachusetts General Hospital",
            "class": "OTHER"
        },
        {
            "name": "Mclean Hospital",
            "class": "OTHER"
        },
        {
            "name": "Massachusetts Institute of Technology",
            "class": "OTHER"
        }
    ]

    Usage: Tracks additional organizations involved in the trial.

    19. brief_summary
    Type: TEXT

    Description: A concise summary of the trial's purpose and objectives.

    Example: "This study aims to evaluate the safety and efficacy of Drug X in adults with Condition Y."

    Usage: Provides a quick overview of the trial.

    20. detailed_description
    Type: TEXT

    Description: A detailed description of the trial, including background, objectives, and methodology.

    Example: "This randomized, double-blind, placebo-controlled study will enroll 200 participants..."

    Usage: Offers in-depth information about the trial.

    21. conditions
    Type: JSONB

    Description: A JSON array or object listing the medical conditions or diseases being studied.

    Example: ["Hypertension", "Diabetes"]

    Usage: Identifies the conditions targeted by the trial.
        """)

        # Chunk 3: Fields 21-30
        vn.train(documentation="""
    22. study_type
    Type: TEXT

    Description: The type of study (e.g., "Interventional", "Observational").

    Example: "Interventional"

    Usage: Categorizes the trial for analysis.

    23. healthy_volunteers
    Type: BOOLEAN

    Description: Indicates whether healthy volunteers are accepted.

    Example: TRUE

    Usage: Specifies eligibility criteria.

    24. phases
    Type: JSONB

    Description: A JSON array or object listing the clinical trial phases (e.g., "PHASE2", "PHASE1").

    Example: ["PHASE2", "PHASE3"]

    Usage: Indicates the trial's phase(s).

    25. allocation
    Type: TEXT

    Description: The method of participant allocation (e.g., "RANDOMIZED", "NON_RANDOMIZED").

    Example: "RANDOMIZED"

    Usage: Describes the trial design.

    26. enrollment_count
    Type: FLOAT

    Description: The number of participants planned or enrolled.

    Example: 200

    Usage: Provides the planned or actual sample size.

    27. eligibility_criteria
    Type: TEXT

    Description: The criteria participants must meet to be eligible for the trial.

    Example: "Ages 18-65, diagnosed with Condition Y, not pregnant."

    Usage: Specifies participant requirements.

    28. gender_based
    Type: BOOLEAN

    Description: Indicates whether the trial is gender-based.

    Example: TRUE

    Usage: Specifies if gender is a criterion.

    29. gender_description
    Type: TEXT

    Description: A description of the gender criteria.

    Example: "Only male participants"

    Usage: Clarifies gender requirements.

    30. sex
    Type: TEXT

    Description: The sex of participants (e.g., "ALL", "MALE", "FEMALE").

    Example: "MALE"

    Usage: Specifies participant sex.

    31. minimum_age
    Type: FLOAT

    Description: The minimum age of participants (in years) (e.g., 18, NULL).

    Example: 18

    Usage: Specifies age requirements.
        """)

        # Chunk 4: Fields 31-38
        vn.train(documentation="""
    32. maximum_age
    Type: FLOAT

    Description: The maximum age of participants (in years) (e.g., 85, NULL).

    Example: 65

    Usage: Specifies age requirements.

    33. std_ages
    Type: JSONB

    Description: A JSON array or object listing standard age groups (e.g., "Adult", "Child").

    Example: ["CHILD", "ADULT", "OLDER_ADULT"]

    Usage: Categorizes participants by age group.

    34. study_population
    Type: TEXT

    Description: A description of the study population.

    Example: "Adults with diagnosed Condition Y"

    Usage: Describes the target population.

    35. sampling_method
    Type: TEXT

    Description: The method used to select participants.

    Example: "Probability Sample"

    Usage: Describes the sampling strategy.

    36. study_references
    Type: JSONB

    Description: A JSON array or object listing references or citations related to the trial.

    Example: [{"pmid": "31467127", "type": "DERIVED", "citation": "Edelmann J, Genomic alterations in high-risk chronic lymphocytic leukemia frequently affect cell cycle key regulators"}, {"pmid": "29063805", "type": "DERIVED", "citation": "Steinbrecher D,Telomere length in poor-risk chronic lymphocytic leukemia"}]

    Usage: Provides related literature.

    37. see_also_links
    Type: JSONB

    Description: A JSON array or object listing related links or resources.

    Example: [{"url": "http://www.tgmg.org", "label": "Tampa General Medical Group"}]

    Usage: Offers additional resources.

    38. avail_ipds
    Type: JSONB

    Description: A JSON array or object indicating the availability of Individual Participant Data (IPD).

    Example: {"available": true, "url": "https://example.com/ipd"}

    Usage: Specifies data sharing policies.

    39. drug_name
    Type: TEXT

    Description: The name of the drug being studied in the clinical trial.

    Example: "Aspirin"

    Usage: Identifies the drug under investigation.

    40. drug_description
    Type: TEXT

    Description: A description of the drug, including its dosing or other relevant details.

    Example: "Preventive treatment: 3X2 EC tablets daily during 2 + 2 months of prevention after run-in of 1 week and intermittent treatment break of 1 week without treatment + 1 month of voluntary follow-up prevention. Acute treatment: 5 X 2 EC tablets daily for max. 10 days per individual vRTI or until symptom resolution."

    Usage: Provides additional information about the drug being studied.
        """)

        # Add training for relevant column selection and ordering
        vn.train(documentation="""
        COLUMN SELECTION AND ORDERING BEST PRACTICES:
        ---------------------------------------------
        - Never use SELECT * in queries. Always select only the relevant columns for the user's question.
        - Place the most important columns first, in an order that is most useful for the user.
        - Never include the 'eligibility_criteria_embedding' column in the SELECT statement unless specifically requested.
        - If unsure, default to the general trial column order above.
        
        MANDATORY COLUMN INCLUSION RULES:
        - ALWAYS include 'eligibility_criteria' when the user asks about eligibility, inclusion criteria, exclusion criteria, participant requirements, or trial requirements
        - ALWAYS include 'drug_name' when the user asks about drugs, medications, treatments, or interventions
        - ALWAYS include 'phases' when the user asks about trial phases or study phases
        - ALWAYS include 'conditions' when the user asks about diseases, medical conditions, or indications
        
        - Example: To show upcoming trials, use:
          SELECT nct_id, brief_title, conditions, drug_name, phases, start_date, completion_date, overall_status, enrollment_count
          FROM clinical_trials
          WHERE start_date >= '{today}'
          ORDER BY start_date ASC
        - Example: For phase 2 diabetes trials:
          SELECT nct_id, brief_title, phases, conditions, start_date, completion_date, drug_name
          FROM clinical_trials
          WHERE phases @> '["PHASE2"]' AND LOWER(conditions::text) LIKE '%diabetes%'
          ORDER BY start_date ASC
        - Example: For similarity search with phase filter:
          SELECT nct_id, brief_title, eligibility_criteria, phases, conditions, drug_name, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
          FROM clinical_trials
          WHERE eligibility_criteria_embedding IS NOT NULL AND phases @> '["PHASE2"]'
          ORDER BY eligibility_criteria_embedding <=> %(embedding)s
        - Never include large text fields (like detailed_description) unless specifically requested.
        - Only use LIMIT if the user specifically asks for a limited number of results (e.g., "top 10", "first 5").
        """)

        # Add specific training for eligibility criteria questions
        vn.train(documentation="""
        ELIGIBILITY CRITERIA QUERY PATTERNS:
        ------------------------------------
        
        CRITICAL RULE: When users ask about eligibility criteria, inclusion criteria, exclusion criteria, 
        participant requirements, or trial requirements, ALWAYS include the 'eligibility_criteria' column in the SELECT statement.
        
        ELIGIBILITY-RELATED KEYWORDS THAT TRIGGER MANDATORY INCLUSION:
        - "eligibility", "eligible", "eligibility criteria"
        - "inclusion criteria", "inclusion requirements", "who can participate"
        - "exclusion criteria", "exclusion requirements", "who cannot participate"
        - "participant requirements", "participation criteria"
        - "trial requirements", "study requirements"
        - "enrollment criteria", "recruitment criteria"
        - "patient criteria", "subject criteria"
        
        MANDATORY COLUMN SET FOR ELIGIBILITY QUESTIONS:
        - nct_id, brief_title, eligibility_criteria (ALWAYS REQUIRED)
        - Additional relevant columns: minimum_age, maximum_age, sex, gender_description, conditions, phases
        
        EXAMPLES OF ELIGIBILITY QUESTIONS AND REQUIRED COLUMNS:
        
        Question: "What are the eligibility criteria for diabetes trials?"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, conditions, minimum_age, maximum_age, sex
             FROM clinical_trials 
             WHERE LOWER(conditions::text) LIKE '%diabetes%' AND eligibility_criteria IS NOT NULL
        
        Question: "Show me inclusion criteria for Phase 2 trials"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, phases, minimum_age, maximum_age, sex
             FROM clinical_trials 
             WHERE phases @> '["PHASE2"]' AND eligibility_criteria IS NOT NULL
        
        Question: "Which trials have specific participant requirements?"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, minimum_age, maximum_age, sex, conditions
             FROM clinical_trials 
             WHERE eligibility_criteria IS NOT NULL
        
        Question: "Find trials with exclusion criteria for pregnant women"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, sex, minimum_age, maximum_age, conditions
             FROM clinical_trials 
             WHERE LOWER(eligibility_criteria) LIKE '%pregnant%' AND eligibility_criteria IS NOT NULL
        
        Question: "What are the enrollment requirements for cancer studies?"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, conditions, minimum_age, maximum_age, sex, phases
             FROM clinical_trials 
             WHERE LOWER(conditions::text) LIKE '%cancer%' AND eligibility_criteria IS NOT NULL
        
        IMPORTANT NOTES:
        - NEVER omit 'eligibility_criteria' when the question involves eligibility, inclusion, exclusion, or participant requirements
        - Always include WHERE eligibility_criteria IS NOT NULL to filter out trials without criteria
        - Include age and sex columns as they are fundamental eligibility parameters
        - Include conditions and phases when relevant to provide context
        """)

        # Add comprehensive similarity search examples to training
        vn.train(documentation="""
        COMPREHENSIVE SIMILARITY SEARCH EXAMPLES:
        ----------------------------------------

        CRITICAL LIMIT RULE FOR SIMILARITY SEARCHES:
        - For all similarity searches using eligibility_criteria_embedding, ALWAYS add 'LIMIT 300' to the SQL query unless the user specifically requests a different limit (e.g., "top 10", "first 5", "limit 1000").
        - If the user specifies a limit, use that value instead of the default 300.
        - If no limit is mentioned, always append 'LIMIT 300' at the end of the SQL.

        The following examples show how to construct similarity queries for various user intents.
        Always include the similarity_score and use the appropriate columns based on the question focus.
        CRITICAL RULE: ALWAYS include the 'drug_name' column in the SELECT statement for all similarity search queries, regardless of the user's question, unless the user specifically requests to exclude it.

        1. BASIC SIMILARITY QUERIES:

        Question: "Find trials with similar eligibility to diabetes patients over 50"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, conditions, minimum_age, maximum_age, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300

        Question: "Show me studies with eligibility like: adults with hypertension, no pregnancy"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, conditions, sex, minimum_age, maximum_age, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300

        2. SIMILARITY + CONDITION FILTERING:

        Question: "Find cancer trials similar to: adults 18-65 with stage 2 cancer"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, conditions, phases, minimum_age, maximum_age, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND LOWER(conditions::text) LIKE '%cancer%'
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300

        Question: "Diabetes studies with eligibility similar to: type 2 diabetes, HbA1c > 7"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, conditions, drug_name, phases, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND LOWER(conditions::text) LIKE '%diabetes%'
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300

        3. SIMILARITY + PHASE FILTERING:
        
        Question: "Phase 2 trials with inclusion criteria like: healthy volunteers, 21-45 years"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, phases, minimum_age, maximum_age, healthy_volunteers, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND phases @> '["PHASE2"]'
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300
        
        Question: "Early phase studies similar to: first-in-human, healthy adults"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, phases, healthy_volunteers, minimum_age, maximum_age, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND phases ?| array['PHASE1', 'PHASE2']
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300

        4. SIMILARITY + STATUS FILTERING:
        
        Question: "Recruiting trials with eligibility similar to: pregnant women, 18-35 years"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, overall_status, sex, minimum_age, maximum_age, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND overall_status = 'RECRUITING'
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300
        
        Question: "Active studies with participant criteria like: elderly patients, dementia"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, overall_status, conditions, minimum_age, maximum_age, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND overall_status IN ('RECRUITING', 'ACTIVE_NOT_RECRUITING', 'ENROLLING_BY_INVITATION')
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300

        5. SIMILARITY + DRUG/INTERVENTION FILTERING:
        
        Question: "Drug trials with eligibility like: treatment-naive patients, first-line therapy"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, drug_description, phases, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND drug_name IS NOT NULL
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300
        
        Question: "Aspirin studies similar to: cardiovascular risk, adults over 40"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, conditions, minimum_age, maximum_age, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND LOWER(drug_name) LIKE '%aspirin%'
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300

        6. SIMILARITY + SPONSOR FILTERING:
        
        Question: "Industry trials with eligibility like: post-marketing surveillance"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, lead_sponsor_name, lead_sponsor_class, phases, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND lead_sponsor_class = 'INDUSTRY'
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300
        
        Question: "NIH studies with participant criteria similar to: rare disease patients"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, lead_sponsor_name, lead_sponsor_class, conditions, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND lead_sponsor_class = 'NIH'
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300

        7. SIMILARITY + DATE FILTERING:
        
        Question: "Recent trials with eligibility similar to: pediatric patients, under 12"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, start_date, minimum_age, maximum_age, std_ages, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND start_date >= '2023-01-01'
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300
        
        Question: "Upcoming studies with inclusion criteria like: women of childbearing potential"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, start_date, sex, minimum_age, maximum_age, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND start_date >= '{today}'
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300

        8. SIMILARITY + AGE FILTERING:
        
        Question: "Trials for seniors similar to: cognitive impairment, over 65"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, minimum_age, maximum_age, std_ages, conditions, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND (minimum_age >= 65 OR std_ages @> '["OLDER_ADULT"]')
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300
        
        Question: "Pediatric studies with eligibility like: children with autism"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, minimum_age, maximum_age, std_ages, conditions, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND (maximum_age <= 18 OR std_ages @> '["CHILD"]')
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300

        9. SIMILARITY + ENROLLMENT SIZE FILTERING:
        
        Question: "Large studies with eligibility similar to: population-based screening"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, enrollment_count, study_population, phases, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND enrollment_count > 1000
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300
        
        Question: "Small pilot studies with criteria like: proof of concept, biomarker"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, enrollment_count, phases, study_type, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND enrollment_count <= 50
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300

        10. COMPLEX MULTI-FILTER SIMILARITY:
        
        Question: "Recruiting Phase 3 cancer trials with eligibility similar to: advanced stage, previous treatment"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, phases, conditions, overall_status, drug_name, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND phases @> '["PHASE3"]'
               AND LOWER(conditions::text) LIKE '%cancer%'
               AND overall_status = 'RECRUITING'
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300
        
        Question: "Industry-sponsored diabetes drug trials similar to: metformin-naive, HbA1c 7-10%"
        SQL: SELECT nct_id, brief_title, eligibility_criteria, drug_name, conditions, lead_sponsor_class, phases, (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
             FROM clinical_trials 
             WHERE eligibility_criteria_embedding IS NOT NULL
               AND LOWER(conditions::text) LIKE '%diabetes%'
               AND lead_sponsor_class = 'INDUSTRY'
               AND drug_name IS NOT NULL
             ORDER BY eligibility_criteria_embedding <=> %(embedding)s
             LIMIT 300
        
        IMPORTANT NOTES FOR ALL SIMILARITY QUERIES:
        - Always include similarity_score: (1 - (eligibility_criteria_embedding <=> %(embedding)s)) AS similarity_score
        - Always include WHERE eligibility_criteria_embedding IS NOT NULL
        - Always ORDER BY eligibility_criteria_embedding <=> %(embedding)s
        - Always add LIMIT 300 unless the user requests a different limit
        - Select columns relevant to the question focus
        - Combine similarity with other filters as needed
        - Use appropriate date placeholders like {today} when referencing current date
        """)

    else:
        st.write("Using previously trained model...")
        
    
    # Always ensure connection is active
    db_host = st.secrets.get("POSTGRES_HOST", "localhost")
    db_port = st.secrets.get("POSTGRES_PORT", 5432)
    db_name = st.secrets.get("POSTGRES_DBNAME")
    db_user = st.secrets.get("POSTGRES_USER")
    db_password = st.secrets.get("POSTGRES_PASSWORD")
    db_sslmode = st.secrets.get("POSTGRES_SSLMODE", "prefer")  # Add this line
    vn.connect_to_postgres(
        host=db_host, 
        port=db_port, 
        dbname=db_name, 
        user=db_user, 
        password=db_password,
        sslmode=db_sslmode  # Add this parameter
    )
      
    return vn

@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    vn = setup_vanna()
    return vn.generate_questions()


@st.cache_data(show_spinner="Generating SQL query ...", ttl=3600)
def generate_sql_cached(question: str, selected_columns=None):
    vn = setup_vanna()

    # Get the current date and format it in a more clear and explicit way
    today = dt_date.today()
    date_str = today.isoformat()

    print(f"Generating SQL for question: {question} (as of {date_str})")
    
    # Enhance the question with date context and column requirements
    enhanced_question = f"{question} (as of {date_str})"
    
    if selected_columns and len(selected_columns) > 0:
        columns_str = ", ".join(selected_columns)
        enhanced_question += f"\n\nIn addition to what columns you judge to be relevant for answering the question, add those: {columns_str}."
    
    return vn.generate_sql(question=enhanced_question, allow_llm_to_see_data=True)

@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str):
    vn = setup_vanna()
    return vn.is_sql_valid(sql=sql)

@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str, question: str = None):
    vn = setup_vanna()
    # Detect if the SQL is a similarity search using eligibility_criteria_embedding
    if "eligibility_criteria_embedding" in sql and "%(embedding)s" in sql:
        # Compute embedding for the question or eligibility text
        if not question:
            raise ValueError("Question text required for similarity search embedding")
        embedding = get_openai_embedding(question)
        if embedding is None:
            raise ValueError("Failed to generate embedding for similarity search")
        
        # Format embedding as string for PostgreSQL vector type
        query_vector_str = f"[{','.join(map(str, embedding))}]"
        
        # Replace the %(embedding)s placeholder with the actual vector string
        # This follows the same approach as query_similar_trials.py
        processed_sql = sql.replace("%(embedding)s", f"'{query_vector_str}'::vector")
        
        # Run the processed SQL
        return vn.run_sql(sql=processed_sql)
    else:
        return vn.run_sql(sql=sql)

@st.cache_data(show_spinner="Checking if we should generate a chart ...")
def should_generate_chart_cached(question, sql, df):
    vn = setup_vanna()
    # Limit sample size to reduce token count
    sample_df = df.head(10) if len(df) > 10 else df
    return vn.should_generate_chart(df=sample_df)

@st.cache_data(show_spinner="Generating Plotly code...")
def generate_plotly_code_cached(question, sql, df):
    vn = setup_vanna()
    sample_df = preprocess_dataframe(df, max_rows=10, max_cols=5)
    try:
        return vn.generate_plotly_code(question=question, sql=sql, df=sample_df)
    except Exception as e:
        st.warning(f"Could not generate visualization: {str(e)}")
        return None


@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    vn = setup_vanna()
    # No need to send the full dataframe to the API for plotting
    return vn.get_plotly_figure(plotly_code=code, df=df)

@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, sql, df):
    vn = setup_vanna()
    # Limit sample size to reduce token count
    sample_df = df.head(5) if len(df) > 5 else df
    return vn.generate_followup_questions(question=question, sql=sql, df=sample_df)

@st.cache_data(show_spinner="Generating summary...")
def generate_summary_cached(question, df):
    vn = setup_vanna()
    
    # Use the preprocessing function but without column limit for summary
    sample_df = preprocess_dataframe(df, max_rows=10, max_cols=len(df.columns))
    
    try:
        return vn.generate_summary(question=question, df=sample_df)
    except Exception as e:
        st.warning(f"Could not generate summary: {str(e)}")
        # Fallback to an even simpler approach if needed
        if "too large" in str(e).lower() or "rate limit" in str(e).lower():
            minimal_df = sample_df.head(3)
            try:
                return vn.generate_summary(question=question, df=minimal_df)
            except:
                return "Summary generation failed due to API limitations. Try a more specific question with a smaller result set."


@st.cache_data(show_spinner="Searching for similar trials...")
def find_similar_trials_cached(query_text: str, limit: int = None):
    """
    Find clinical trials with eligibility criteria similar to the query text.
    
    Args:
        query_text (str): The text to search for similar eligibility criteria
        limit (int): Maximum number of trials to return (optional, no limit if None)
    
    Returns:
        pandas.DataFrame: DataFrame containing similar trials with similarity scores
    """
    # Generate the similarity search SQL
    limit_clause = f" LIMIT {limit}" if limit is not None else ""
    similarity_sql = f"""
    SELECT 
        nct_id,
        brief_title,
        eligibility_criteria,
        conditions,
        phases,
        overall_status,
        start_date,
        completion_date,
        enrollment_count,
        1 - (eligibility_criteria_embedding <=> %(embedding)s) AS similarity_score
    FROM clinical_trials
    WHERE eligibility_criteria_embedding IS NOT NULL
    ORDER BY eligibility_criteria_embedding <=> %(embedding)s{limit_clause}
    """
    
    # Use the existing run_sql_cached function which handles embedding generation
    return run_sql_cached(sql=similarity_sql, question=query_text)

@st.cache_data(show_spinner="Checking embedding availability...")
def check_embeddings_available():
    """
    Check if there are clinical trials with embeddings available in the database.
    
    Returns:
        dict: Dictionary with count of trials with embeddings and total trials
    """
    vn = setup_vanna()
    
    # Check total trials
    total_result = vn.run_sql("SELECT COUNT(*) as total_trials FROM clinical_trials")
    total_trials = total_result.iloc[0]['total_trials'] if not total_result.empty else 0
    
    # Check trials with embeddings
    embedding_result = vn.run_sql("""
        SELECT COUNT(*) as trials_with_embeddings 
        FROM clinical_trials 
        WHERE eligibility_criteria_embedding IS NOT NULL
    """)
    trials_with_embeddings = embedding_result.iloc[0]['trials_with_embeddings'] if not embedding_result.empty else 0
    
    return {
        'total_trials': total_trials,
        'trials_with_embeddings': trials_with_embeddings,
        'embedding_coverage': trials_with_embeddings / total_trials if total_trials > 0 else 0
    }



