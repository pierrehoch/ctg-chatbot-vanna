import streamlit as st
import os
# from vanna.rag import VannaRAG
from vanna.chromadb import ChromaDB_VectorStore
from vanna.openai import OpenAI_Chat
from datetime import date as dt_date, timedelta
import pandas as pd
import numpy as np


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
            
            print(f"Trying direct psycopg2 connection...")
            
            # Test connection with psycopg2 directly
            test_conn = psycopg2.connect(**conn_params)
            test_conn.close()
            print("Direct psycopg2 connection successful!")
            
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
        """)
        
      
        # Time-based queries and best practices
        vn.train(documentation="""
        TIME-BASED QUERY EXAMPLES AND BEST PRACTICES:
        ---------------------------------------------
        
        - To find the soonest upcoming trials (trials starting after today, ordered by start date):
        
          SQL: 
          SELECT * FROM clinical_trials 
          WHERE start_date >= '{today}' 
          ORDER BY start_date ASC 
          LIMIT 10
        
        - To find trials that are currently recruiting and will start soon:
        
          SQL:
          SELECT * FROM clinical_trials 
          WHERE overall_status = 'RECRUITING' 
            AND start_date >= '{today}' 
          ORDER BY start_date ASC 
          LIMIT 10
        
        - To find trials that are about to complete soon:
        
          SQL:
          SELECT * FROM clinical_trials 
          WHERE completion_date >= '{today}' 
          ORDER BY completion_date ASC 
          LIMIT 10
        
        - To find trials that started in the past month:
        
          SQL:
          SELECT * FROM clinical_trials 
          WHERE start_date >= '{one_month_ago}' 
            AND start_date < '{today}'
        
        - To find trials that are ongoing as of today:
        
          SQL:
          SELECT * FROM clinical_trials 
          WHERE start_date <= '{today}' 
            AND (completion_date IS NULL OR completion_date >= '{today}')
        
        - Always use the date provided in the question (e.g., "as of 2025-06-23") as the reference for 'today'.
        - Use ORDER BY with ASC for soonest/upcoming, DESC for most recent/past.
        - Use LIMIT to restrict the number of results for summary or list questions.
        
        Replace {today} with the current date (provided in the prompt).
        Replace {one_month_ago} with the date one month before today.
        
        EXAMPLES:
        ---------
        Question: "What are the soonest upcoming trials?"
        SQL: SELECT * FROM clinical_trials WHERE start_date >= '{today}' ORDER BY start_date ASC LIMIT 10
        
        Question: "Which trials are about to complete?"
        SQL: SELECT * FROM clinical_trials WHERE completion_date >= '{today}' ORDER BY completion_date ASC LIMIT 10
        
        Question: "Show me the most recently started trials"
        SQL: SELECT * FROM clinical_trials WHERE start_date <= '{today}' ORDER BY start_date DESC LIMIT 10
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
        SQL: SELECT * FROM clinical_trials WHERE phases @> '["PHASE2"]'
        
        Question: "Find Phase 1 or Phase 2 trials"  
        SQL: SELECT * FROM clinical_trials WHERE phases @> '["PHASE1"]' OR phases @> '["PHASE2"]'
        
        Question: "Trials in early phases"
        SQL: SELECT * FROM clinical_trials WHERE phases ?| array['PHASE1', 'PHASE2']
        
        Question: "Phase 3 trials for diabetes"
        SQL: SELECT * FROM clinical_trials WHERE phases @> '["PHASE3"]' AND LOWER(conditions::text) LIKE '%diabetes%'
        
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

        3. brief_title
        Type: TEXT

        Description: A short, descriptive title for the clinical trial.

        Example: "Efficacy of Drug X in Treating Condition Y"

        Usage: Provides a quick reference for the trial's purpose.

        4. official_title
        Type: TEXT

        Description: The formal, official title of the clinical trial.

        Example: "A Randomized, Double-Blind, Placebo-Controlled Study of Drug X in Patients with Condition Y"

        Usage: Used in official documentation and regulatory submissions.

        5. overall_status
        Type: TEXT

        Description: The current status of the clinical trial (e.g., "RECRUITING", "COMPLETED", "NOT_YET_RECRUITING", "UNKNOWN", "WITHDRAWN", "TERMINATED").

        Example: "RECRUITING"

        Usage: Indicates the trial's current phase or stage.

        6. start_date
        Type: DATE

        Description: The date when the trial started.

        Example: 2023-01-15

        Usage: Provides timeline information for the trial.

        7. start_date_type
        Type: TEXT

        Description: The type of start date (e.g., "Actual", "Anticipated").

        Example: "Actual"

        Usage: Clarifies whether the date is confirmed or estimated.

        8. primary_completion_date
        Type: DATE

        Description: The date when the primary outcome measure is assessed.

        Example: 2024-06-30

        Usage: Marks a key milestone in the trial timeline.

        9. primary_completion_date_type
        Type: TEXT

        Description: The type of primary completion date (e.g., "Actual", "Anticipated").

        Example: "Actual"

        Usage: Clarifies whether the date is confirmed or estimated.

        10. completion_date
        Type: DATE

        Description: The date when the trial is completed.

        Example: 2024-12-31

        Usage: Indicates the end of the trial.
            """)

        # Chunk 2: Fields 11-20
        vn.train(documentation="""
    11. completion_date_type
    Type: TEXT

    Description: The type of completion date (e.g., "Actual", "Anticipated").

    Example: "Anticipated"

    Usage: Clarifies whether the date is confirmed or estimated.

    12. study_first_submit_date
    Type: DATE

    Description: The date when the trial was first submitted to ClinicalTrials.gov.

    Example: 2022-11-10

    Usage: Tracks the submission timeline.

    13. last_update_date
    Type: DATE

    Description: The date when the trial record was last updated.

    Example: 2024-03-05

    Usage: Indicates the most recent update to the trial information.

    14. last_update_date_type
    Type: TEXT

    Description: The type of last update date (e.g., "Actual", "Anticipated").

    Example: "Actual"

    Usage: Clarifies whether the date is confirmed or estimated.

    15. lead_sponsor_name
    Type: TEXT

    Description: The name of the primary sponsor of the clinical trial.

    Example: "National Institutes of Health"

    Usage: Identifies the main organization responsible for the trial.

    16. lead_sponsor_class
    Type: TEXT

    Description: The classification of the lead sponsor (e.g., "NIH", "Industry", "Other").

    Example: "NIH"

    Usage: Categorizes the sponsor for analysis and reporting.

    17. collaborators
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

    18. brief_summary
    Type: TEXT

    Description: A concise summary of the trial's purpose and objectives.

    Example: "This study aims to evaluate the safety and efficacy of Drug X in adults with Condition Y."

    Usage: Provides a quick overview of the trial.

    19. detailed_description
    Type: TEXT

    Description: A detailed description of the trial, including background, objectives, and methodology.

    Example: "This randomized, double-blind, placebo-controlled study will enroll 200 participants..."

    Usage: Offers in-depth information about the trial.

    20. conditions
    Type: JSONB

    Description: A JSON array or object listing the medical conditions or diseases being studied.

    Example: ["Hypertension", "Diabetes"]

    Usage: Identifies the conditions targeted by the trial.
        """)

        # Chunk 3: Fields 21-30
        vn.train(documentation="""
    21. study_type
    Type: TEXT

    Description: The type of study (e.g., "Interventional", "Observational").

    Example: "Interventional"

    Usage: Categorizes the trial for analysis.

    22. healthy_volunteers
    Type: BOOLEAN

    Description: Indicates whether healthy volunteers are accepted.

    Example: TRUE

    Usage: Specifies eligibility criteria.

    23. phases
    Type: JSONB

    Description: A JSON array or object listing the clinical trial phases (e.g., "PHASE2", "PHASE1").

    Example: ["PHASE2", "PHASE3"]

    Usage: Indicates the trial's phase(s).

    24. allocation
    Type: TEXT

    Description: The method of participant allocation (e.g., "RANDOMIZED", "NON_RANDOMIZED").

    Example: "RANDOMIZED"

    Usage: Describes the trial design.

    25. enrollment_count
    Type: FLOAT

    Description: The number of participants planned or enrolled.

    Example: 200

    Usage: Provides the planned or actual sample size.

    26. eligibility_criteria
    Type: TEXT

    Description: The criteria participants must meet to be eligible for the trial.

    Example: "Ages 18-65, diagnosed with Condition Y, not pregnant."

    Usage: Specifies participant requirements.

    27. gender_based
    Type: BOOLEAN

    Description: Indicates whether the trial is gender-based.

    Example: TRUE

    Usage: Specifies if gender is a criterion.

    28. gender_description
    Type: TEXT

    Description: A description of the gender criteria.

    Example: "Only male participants"

    Usage: Clarifies gender requirements.

    29. sex
    Type: TEXT

    Description: The sex of participants (e.g., "ALL", "MALE", "FEMALE").

    Example: "MALE"

    Usage: Specifies participant sex.

    30. minimum_age
    Type: FLOAT

    Description: The minimum age of participants (in years) (e.g., 18, NULL).

    Example: 18

    Usage: Specifies age requirements.
        """)

        # Chunk 4: Fields 31-38
        vn.train(documentation="""
    31. maximum_age
    Type: FLOAT

    Description: The maximum age of participants (in years) (e.g., 85, NULL).

    Example: 65

    Usage: Specifies age requirements.

    32. std_ages
    Type: JSONB

    Description: A JSON array or object listing standard age groups (e.g., "Adult", "Child").

    Example: ["CHILD", "ADULT", "OLDER_ADULT"]

    Usage: Categorizes participants by age group.

    33. study_population
    Type: TEXT

    Description: A description of the study population.

    Example: "Adults with diagnosed Condition Y"

    Usage: Describes the target population.

    34. sampling_method
    Type: TEXT

    Description: The method used to select participants.

    Example: "Probability Sample"

    Usage: Describes the sampling strategy.

    35. study_references
    Type: JSONB

    Description: A JSON array or object listing references or citations related to the trial.

    Example: [{"pmid": "31467127", "type": "DERIVED", "citation": "Edelmann J, Genomic alterations in high-risk chronic lymphocytic leukemia frequently affect cell cycle key regulators"}, {"pmid": "29063805", "type": "DERIVED", "citation": "Steinbrecher D,Telomere length in poor-risk chronic lymphocytic leukemia"}]

    Usage: Provides related literature.

    36. see_also_links
    Type: JSONB

    Description: A JSON array or object listing related links or resources.

    Example: [{"url": "http://www.tgmg.org", "label": "Tampa General Medical Group"}]

    Usage: Offers additional resources.

    37. avail_ipds
    Type: JSONB

    Description: A JSON array or object indicating the availability of Individual Participant Data (IPD).

    Example: {"available": true, "url": "https://example.com/ipd"}

    Usage: Specifies data sharing policies.

    38. drug_name
    Type: TEXT

    Description: The name of the drug being studied in the clinical trial.

    Example: "Aspirin"

    Usage: Identifies the drug under investigation.

    39. drug_description
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
        - For general trial queries, prefer this column order:
            nct_id, brief_title, conditions, drug_name, phases, start_date, completion_date, overall_status, enrollment_count
        - For queries about drugs, use: drug_name, drug_description, nct_id, brief_title, overall_status, start_date, completion_date
        - For queries about sponsors/collaborators, use: lead_sponsor_name, collaborators, nct_id, brief_title, overall_status, start_date
        - For queries about eligibility, use: nct_id, brief_title, eligibility_criteria, minimum_age, maximum_age, sex, gender_description
        - For queries about study population, use: nct_id, brief_title, study_population, std_ages, enrollment_count
        - Always omit columns that are not relevant to the user's question.
        - If unsure, default to the general trial column order above.
        - Example: To show upcoming trials, use:
          SELECT nct_id, brief_title, conditions, drug_name, phases, start_date, completion_date, overall_status, enrollment_count
          FROM clinical_trials
          WHERE start_date >= '{today}'
          ORDER BY start_date ASC
          LIMIT 10
        - Example: For phase 2 diabetes trials:
          SELECT nct_id, brief_title, phases, conditions, start_date, completion_date, drug_name
          FROM clinical_trials
          WHERE phases @> '["PHASE2"]' AND LOWER(conditions::text) LIKE '%diabetes%'
          ORDER BY start_date ASC
        - Never include large text fields (like detailed_description) unless specifically requested.
        """)

        # Add training for user-selected columns
        # vn.train(documentation="""
        # USER COLUMN SELECTION HANDLING:
        # ------------------------------
        # When user-selected columns are provided, ALWAYS include them in the SELECT statement.
        
        # Rules for column selection:
        # 1. If user has selected specific columns, include ALL of them in the SELECT statement
        # 2. Add other relevant columns based on the question, but user-selected columns take priority
        # 3. Always include user-selected columns even if they seem irrelevant to the question
        # 4. Order columns with user-selected ones first, then add other relevant columns
        # 5. Never exclude user-selected columns from the query
        
        # Example:
        # - User selected columns: ["nct_id", "drug_name", "enrollment_count"]
        # - Question: "Show me diabetes trials"
        # - SQL should start with: SELECT nct_id, drug_name, enrollment_count, conditions, ...
        
        # The user-selected columns should ALWAYS appear in the SELECT clause regardless of the question.
        # """)

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
        enhanced_question += f"\n\In addition to what columns you judge to be relevant for answering the question, add those: {columns_str}."
    
    return vn.generate_sql(question=enhanced_question, allow_llm_to_see_data=True)

@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str):
    vn = setup_vanna()
    return vn.is_sql_valid(sql=sql)

@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    vn = setup_vanna()
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


