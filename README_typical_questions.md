# Typical Questions File

This file contains a curated list of typical questions that users might ask the Elevio CTG Assistant. The application randomly selects 3 questions from this file to display as suggestions when no question is currently active.

## Format

Each line in the file should contain one question. Empty lines are ignored.

## Usage

The questions are loaded by the `get_random_questions()` function in `src/vanna_calls.py` and cached using Streamlit's caching mechanism.

## Adding New Questions

To add new questions:
1. Add each question on a new line
2. Keep questions concise and specific to clinical trials
3. Focus on common use cases like:
   - Finding trials by phase
   - Searching by condition
   - Eligibility criteria queries
   - Time-based searches
   - Enrollment information

## Examples of Good Questions

- "What are the soonest upcoming trials?"
- "Show me Phase 2 trials for diabetes"
- "Find trials recruiting participants now"
- "Show me eligibility criteria for cancer trials"
- "Find trials with enrollment over 500 participants"

The questions should be phrased naturally as users would ask them in conversation with the assistant.
