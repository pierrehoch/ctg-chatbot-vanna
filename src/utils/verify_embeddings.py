import numpy as np
from get_embedding import get_openai_embedding

def verify_embedding(text="This is a test sentence to verify embeddings are working properly."):
    """Verify that embeddings are created correctly for the given text."""
    print(f"Generating embedding for text: '{text}'")
    
    embedding = get_openai_embedding(text)
    
    if embedding is None:
        print("❌ Failed to generate embedding. Check your API key and connection.")
        return False
    
    # Verify embedding properties
    embedding_array = np.array(embedding)
    embedding_length = len(embedding)
    
    print(f"✅ Successfully generated embedding!")
    print(f"Embedding length: {embedding_length}")
    print(f"First 5 values: {embedding[:5]}")
    print(f"Embedding shape: {embedding_array.shape}")
    print(f"Embedding type: {type(embedding)}")
    
    # Check for NaN values
    if np.isnan(embedding_array).any():
        print("⚠️ Warning: Embedding contains NaN values")
    
    # Check embedding vector normalization (should be close to 1.0 for normalized embeddings)
    norm = np.linalg.norm(embedding_array)
    print(f"L2 norm of embedding: {norm:.6f}")
    
    return True

if __name__ == "__main__":
    print("=== OpenAI Embedding Verification ===")
    
    # Test with default text
    default_success = verify_embedding()
    
    # Test with custom text if provided
    import sys
    if len(sys.argv) > 1:
        custom_text = " ".join(sys.argv[1:])
        print("\n=== Testing with custom input ===")
        verify_embedding(custom_text)
        
    print("\nRun this script with custom text: python verify_embeddings.py 'Your custom text here'")
