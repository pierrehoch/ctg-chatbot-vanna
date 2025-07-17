import os
import random

def get_random_questions(num_questions=3):
    """
    Load typical questions from file and return a random selection.
    
    Args:
        num_questions (int): Number of questions to return (default: 3)
    
    Returns:
        list: List of randomly selected questions
    """
    try:
        # Get the path to the typical questions file
        questions_file = os.path.join(os.path.dirname(__file__), 'typical_questions.txt')
        
        # Read questions from file
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f.readlines() if line.strip()]
        
        # Return random selection
        if len(questions) >= num_questions:
            return random.sample(questions, num_questions)
        else:
            return questions  # Return all if fewer than requested
    except FileNotFoundError:
        # Fallback questions if file doesn't exist
        return [
            "What are the soonest upcoming trials?",
            "Show me Phase 2 trials for diabetes",
            "Find trials recruiting participants now"
        ]
    except Exception as e:
        print(f"Error loading questions: {str(e)}")
        return []

# Test the function
print("Testing get_random_questions function:")
questions = get_random_questions(3)
print(f"Got {len(questions)} questions:")
for i, q in enumerate(questions, 1):
    print(f"{i}. {q}")
