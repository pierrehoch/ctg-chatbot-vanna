try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import time
import streamlit as st
from src.vanna_calls import (
    generate_questions_cached,
    generate_sql_cached,
    run_sql_cached,
    generate_plotly_code_cached,
    generate_plot_cached,
    generate_followup_cached,
    should_generate_chart_cached,
    is_sql_valid_cached,
    generate_summary_cached
)

st.set_page_config(initial_sidebar_state="collapsed", page_title="Elevio CTG Assistant", page_icon=":robot_face:", layout="wide")

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["USER_KEY"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
    st.sidebar.title("Output Settings")
    st.sidebar.checkbox("Show SQL", value=True, key="show_sql")
    st.sidebar.checkbox("Show Table", value=True, key="show_table")
    st.sidebar.checkbox("Enable SQL Editor", value=True, key="enable_sql_editor", help="Allow editing of generated SQL queries before execution")
    # st.sidebar.checkbox("Show Plotly Code", value=True, key="show_plotly_code")
    # st.sidebar.checkbox("Show Chart", value=True, key="show_chart")
    st.sidebar.checkbox("Show Summary", value=False, key="show_summary")
    # st.sidebar.checkbox("Show Follow-up Questions", value=True, key="show_followup")
    
    # Add column selection widget
    st.sidebar.title("Column Selection")
    st.sidebar.write("Specific columns to add to the answers:")
    
    # Define all available columns
    all_columns = [
        "nct_id", "org_study_id", "brief_title", "official_title", "overall_status",
        "start_date", "start_date_type", "primary_completion_date", "primary_completion_date_type",
        "completion_date", "completion_date_type", "study_first_submit_date",
        "last_update_date", "last_update_date_type", "lead_sponsor_name",
        "lead_sponsor_class", "collaborators", "brief_summary", "detailed_description",
        "conditions", "study_type", "phases", "allocation", "enrollment_count",
        "eligibility_criteria", "healthy_volunteers", "gender_based", "gender_description",
        "sex", "minimum_age", "maximum_age", "std_ages", "study_population",
        "sampling_method", "study_references", "see_also_links", "avail_ipds",
        "drug_name", "drug_description"
    ]
    
    # Default selected columns (most commonly useful)
    default_columns = []
    
    # Create checkboxes for each column
    with st.sidebar.container():
        selected_columns = []
        for column in all_columns:
            is_default = column in default_columns
            if st.checkbox(column, value=is_default, key=f"col_{column}"):
                selected_columns.append(column)
    
    
    st.title("Elevio CTG Assistant")
    # st.sidebar.write(st.session_state)
    st.button("New Question", on_click=lambda: set_question(None),)
    
    
    def set_question(question):
        st.session_state["my_question"] = question
        # Clear SQL editor state when asking a new question
        if "editable_sql" in st.session_state:
            del st.session_state["editable_sql"]
        if "sql_executed" in st.session_state:
            del st.session_state["sql_executed"]
        if "is_editing_sql" in st.session_state:
            del st.session_state["is_editing_sql"]
    
    
    # assistant_message_suggested = st.chat_message(
    #     "assistant", avatar="🤖"
    # )
    # if assistant_message_suggested.button("Click to show suggested questions"):
    #     st.session_state["my_question"] = None
    #     questions = generate_questions_cached()
    #     for i, question in enumerate(questions):
    #         time.sleep(0.05)
    #         button = st.button(
    #             question,
    #             on_click=set_question,
    #             args=(question,),
    #         )
    
    my_question = st.session_state.get("my_question", default=None)
    
    if my_question is None:
        my_question = st.chat_input(
            "Ask me a question about your data",
        )
    
    
    if my_question:
        st.session_state["my_question"] = my_question
        user_message = st.chat_message("user")
        user_message.write(f"{my_question}")        # Pass selected columns to SQL generation
        sql = generate_sql_cached(question=my_question, selected_columns=selected_columns)

        if sql:
            if is_sql_valid_cached(sql=sql):
                # Initialize the editable SQL in session state if not present
                if "editable_sql" not in st.session_state:
                    st.session_state["editable_sql"] = sql
                
                if st.session_state.get("show_sql", True):
                    assistant_message_sql = st.chat_message(
                        "assistant", avatar="🤖"
                    )
                    
                    if st.session_state.get("enable_sql_editor", True):
                        # Define all available columns from the clinical_trials table
                        all_columns = [
                            "nct_id", "org_study_id", "brief_title", "official_title", 
                            "overall_status", "start_date", "start_date_type", "primary_completion_date", 
                            "primary_completion_date_type", "completion_date", "completion_date_type", 
                            "study_first_submit_date", "last_update_date", "last_update_date_type", 
                            "lead_sponsor_name", "lead_sponsor_class", "collaborators", "brief_summary", 
                            "detailed_description", "conditions", "study_type", "healthy_volunteers", 
                            "phases", "allocation", "enrollment_count", "eligibility_criteria", 
                            "eligibility_criteria_embedding", "gender_based", "gender_description", 
                            "sex", "minimum_age", "maximum_age", "std_ages", "study_population", 
                            "sampling_method", "study_references", "see_also_links", "avail_ipds", 
                            "drug_name", "drug_description"
                        ]
                        
                        # Initialize editing state if not present
                        if "is_editing_sql" not in st.session_state:
                            st.session_state["is_editing_sql"] = False
                        
                        with assistant_message_sql:
                            if not st.session_state["is_editing_sql"]:
                                # Display mode: Show pretty code block with Edit button
                                st.write("**Generated SQL Query:**")
                                st.code(st.session_state["editable_sql"], language="sql", line_numbers=True)
                                
                                col1, col2, col3 = st.columns([1, 1, 2])
                                with col1:
                                    if st.button("✏️ Edit SQL"):
                                        st.session_state["is_editing_sql"] = True
                                        st.rerun()
                                
                                with col2:
                                    run_query = st.button("🚀 Run Query", type="primary")
                                
                                with col3:
                                    st.write("")  # Empty space for alignment
                                
                                edited_sql = st.session_state["editable_sql"]
                            
                            else:
                                # Edit mode: Show editable text area with Save/Cancel buttons
                                st.write("**Edit SQL Query:**")
                                
                                # Add some styling for the text area
                                st.markdown(
                                    """<style>
                                    .stTextArea textarea {
                                        font-size: 20px !important;
                                        background-color: #18212f !important;
                                        border: 1px solid #ABB0B5 !important;
                                        border-radius: 8px !important;
                                    }
                                    </style>""", 
                                    unsafe_allow_html=True
                                )
                                
                                # Function to extract columns from SELECT clause
                                def extract_select_columns(sql_text):
                                    import re
                                    # Find the SELECT clause (case insensitive)
                                    select_pattern = r'SELECT\s+(.*?)\s+FROM'
                                    match = re.search(select_pattern, sql_text.upper())
                                    if not match:
                                        return []
                                    
                                    select_clause = match.group(1)
                                    # Split by comma and clean up column names
                                    columns = []
                                    for col in select_clause.split(','):
                                        col = col.strip()
                                        # Remove aliases (AS alias_name)
                                        col = re.sub(r'\s+AS\s+\w+', '', col, flags=re.IGNORECASE)
                                        # Extract just the column name (remove table prefixes, functions, etc.)
                                        col_match = re.search(r'(\w+)(?:\s*$|\s*,)', col)
                                        if col_match and col_match.group(1).lower() not in ['select', 'from', 'where', 'order', 'by', 'limit', 'as']:
                                            columns.append(col_match.group(1).lower())
                                    
                                    return columns
                                
                                # Get currently selected columns from the SQL
                                current_columns = extract_select_columns(st.session_state["editable_sql"])
                                
                                # Available columns that are not already in the SELECT
                                available_to_add = [col for col in all_columns if col.lower() not in current_columns]
                                
                                # Column addition dropdown
                                if available_to_add:
                                    col_add1, col_add2 = st.columns([2, 1])
                                    with col_add1:
                                        selected_column_to_add = st.selectbox(
                                            "Add column to SELECT:",
                                            options=[""] + available_to_add,
                                            help="Select a column to add to the SELECT clause"
                                        )
                                    with col_add2:
                                        if st.button("➕ Add Column") and selected_column_to_add:
                                            # Add the column to the SELECT clause
                                            sql_lines = st.session_state["editable_sql"].split('\n')
                                            for i, line in enumerate(sql_lines):
                                                if 'SELECT' in line.upper():
                                                    # Find the end of the SELECT clause
                                                    select_end = i
                                                    for j in range(i, len(sql_lines)):
                                                        if 'FROM' in sql_lines[j].upper():
                                                            select_end = j - 1
                                                            break
                                                    
                                                    # Add the column to the last line of SELECT clause
                                                    if select_end < len(sql_lines):
                                                        # Remove trailing comma or add comma if needed
                                                        last_select_line = sql_lines[select_end].rstrip()
                                                        if not last_select_line.endswith(','):
                                                            last_select_line += ','
                                                        sql_lines[select_end] = last_select_line
                                                        sql_lines.insert(select_end + 1, f"        {selected_column_to_add}")
                                                    break
                                            
                                            st.session_state["editable_sql"] = '\n'.join(sql_lines)
                                            st.rerun()
                                
                                edited_sql = st.text_area(
                                    "",
                                    value=st.session_state["editable_sql"],
                                    height=250,
                                    help="Edit the SQL query and click Save to apply changes.",
                                    key="sql_editor",
                                    label_visibility="collapsed",
                                    placeholder="SQL query will appear here..."
                                )
                                
                                col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
                                
                                with col1:
                                    if st.button("💾 Save"):
                                        st.session_state["editable_sql"] = edited_sql
                                        st.session_state["is_editing_sql"] = False
                                        st.rerun()
                                
                                with col2:
                                    if st.button("❌ Cancel"):
                                        st.session_state["is_editing_sql"] = False
                                        st.rerun()
                                
                                with col3:
                                    if st.button("🔄 Reset"):
                                        st.session_state["editable_sql"] = sql
                                        st.session_state["is_editing_sql"] = False
                                        st.rerun()
                                
                                with col4:
                                    run_query = st.button("🚀 Run Query", type="primary")
                    else:
                        # Just show the SQL without editing capability
                        assistant_message_sql.write("**Generated SQL Query:**")
                        assistant_message_sql.code(sql, language="sql", line_numbers=True)
                        run_query = True  # Auto-run when editor is disabled
                        edited_sql = sql
                
                # SQL execution logic
                if st.session_state.get("enable_sql_editor", True):
                    # Only run the query if the button is clicked or if this is the first time
                    if run_query or "sql_executed" not in st.session_state:
                        # Validate the edited SQL
                        if is_sql_valid_cached(sql=edited_sql):
                            st.session_state["sql_executed"] = True
                            df = run_sql_cached(sql=edited_sql, question=my_question)
                        else:
                            st.error("❌ The modified SQL query is not valid. Please check your syntax.")
                            st.stop()
                    else:
                        # Don't execute anything if the button wasn't clicked
                        # st.info("👆 Click 'Run Query' to execute the SQL query above.")
                        st.stop()
                else:
                    # SQL editor is disabled, run the query directly
                    df = run_sql_cached(sql=sql, question=my_question)
            else:
                assistant_message = st.chat_message(
                    "assistant", avatar="🤖"
                )
                assistant_message.write(sql)
                st.stop()
    
            if df is not None:
                st.session_state["df"] = df
    
            if st.session_state.get("df") is not None:
                if st.session_state.get("show_table", True):
                    df = st.session_state.get("df")
                    assistant_message_table = st.chat_message(
                        "assistant",
                        avatar="🤖",
                    )
                    
                    assistant_message_table.dataframe(df)
    
                if False & should_generate_chart_cached(question=my_question, sql=sql, df=df):
    
                    code = generate_plotly_code_cached(question=my_question, sql=sql, df=df)
    
                    if st.session_state.get("show_plotly_code", False):
                        assistant_message_plotly_code = st.chat_message(
                            "assistant",
                            avatar="🤖",
                        )
                        assistant_message_plotly_code.code(
                            code, language="python", line_numbers=True
                        )
    
                    if code is not None and code != "":
                        if st.session_state.get("show_chart", True):
                            assistant_message_chart = st.chat_message(
                                "assistant",
                                avatar="🤖",
                            )
                            fig = generate_plot_cached(code=code, df=df)
                            if fig is not None:
                                assistant_message_chart.plotly_chart(fig)
                            else:
                                assistant_message_chart.error("I couldn't generate a chart")
    
                if st.session_state.get("show_summary", True):
                    assistant_message_summary = st.chat_message(
                        "assistant",
                        avatar="🤖",
                    )
                    summary = generate_summary_cached(question=my_question, df=df)
                    if summary is not None:
                        assistant_message_summary.text(summary)
    
                if False & st.session_state.get("show_followup", True):
                    assistant_message_followup = st.chat_message(
                        "assistant",
                        avatar="🤖",
                    )
                    followup_questions = generate_followup_cached(
                        question=my_question, sql=sql, df=df
                    )
                    st.session_state["df"] = None
    
                    if len(followup_questions) > 0:
                        assistant_message_followup.text(
                            "Here are some possible follow-up questions"
                        )
                        # Print the first 5 follow-up questions
                        for question in followup_questions[:5]:
                            assistant_message_followup.button(question, on_click=set_question, args=(question,))
    
        else:
            assistant_message_error = st.chat_message(
                "assistant", avatar="🤖"
            )
            assistant_message_error.error("I wasn't able to generate SQL for that question")
