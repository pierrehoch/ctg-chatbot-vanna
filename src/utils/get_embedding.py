from openai import OpenAI
import streamlit as st
import os
import time
from tqdm import tqdm
# from .logging_config import get_logger, ProgressTracker

def get_openai_embedding(text):
    """Compute OpenAI embedding for a given text using the API key from secrets or environment."""
    # Handle empty or None values
    if text is None or not isinstance(text, str) or text.strip() == "":
        return None
        
    try:
        # Try to get API key from Streamlit secrets first, then from environment variables
        api_key = None
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            # Not in Streamlit context or secrets not available
            api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in Streamlit secrets or environment variables")
        
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        
        embedding = response.data[0].embedding
        
        # Validate the embedding
        if not isinstance(embedding, list) or len(embedding) == 0:
            print(f"Warning: Invalid embedding format received for text: {text[:100]}...")
            return None
        
        # Validate embedding dimension (text-embedding-3-small should produce 1536 dimensions)
        if len(embedding) != 1536:
            print(f"Warning: Unexpected embedding dimension {len(embedding)}, expected 1536")
            
        # Ensure all values are floats
        try:
            embedding = [float(x) for x in embedding]
        except (ValueError, TypeError) as e:
            print(f"Error converting embedding to floats: {e}")
            return None
            
        return embedding
        
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def chunk_texts(texts, batch_size):
    """Generator to chunk texts into batches."""
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]


def get_openai_embeddings_batch(texts, batch_size=500, model="text-embedding-3-small"):
    """
    Compute OpenAI embeddings for a list of texts using batching with rate limit handling.
    
    Args:
        texts (list): List of text strings to generate embeddings for
        batch_size (int): Size of each batch to send to the API (default 500 for faster processing)
        model (str): OpenAI model to use for embeddings
    
    Returns:
        list: List of embeddings corresponding to input texts (None for failed embeddings)
    
    Note:
        - Default batch_size increased to 500 for better performance
        - OpenAI API can handle up to 8,192 tokens per request
        - Automatic token estimation helps prevent API limit errors
        - For very large texts, batch size will be dynamically reduced
    """
    # Handle empty input
    if not texts:
        return []
    
    # Get API key
    logger = get_logger()
    try:
        api_key = None
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except Exception:
            api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in Streamlit secrets or environment variables")
        
        client = OpenAI(api_key=api_key)
        
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {e}")
        return [None] * len(texts)
    
    # Preprocess texts - filter out empty/None texts but keep track of original indices
    processed_texts = []
    text_indices = []
    
    for i, text in enumerate(texts):
        if text is not None and isinstance(text, str) and text.strip() != "":
            processed_texts.append(text.strip())
            text_indices.append(i)
    
    logger.info(f"Processing {len(processed_texts)} valid texts out of {len(texts)} total texts")
    
    if not processed_texts:
        logger.warning("No valid texts found for embedding generation")
        return [None] * len(texts)
    
    # Estimate tokens and adjust batch size if needed
    avg_text_length = sum(len(text.split()) for text in processed_texts[:10]) / min(10, len(processed_texts))
    estimated_tokens_per_text = avg_text_length * 1.2  # Less conservative token estimation
    
    # Dynamically adjust batch size to stay under token limit per request
    max_tokens_per_request = 8000  # Increased token limit
    if estimated_tokens_per_text > 0:
        optimal_batch_size = int(max_tokens_per_request / estimated_tokens_per_text)
        dynamic_batch_size = min(batch_size, max(optimal_batch_size, 50))  # Increased minimum to 50
        
        if dynamic_batch_size != batch_size:
            logger.info(f"Adjusted batch size from {batch_size} to {dynamic_batch_size} based on text length")
            logger.debug(f"Estimated tokens per text: {estimated_tokens_per_text:.1f}")
            batch_size = dynamic_batch_size
    
    # Initialize results array
    embeddings = [None] * len(texts)
    
    # Process in batches with progress tracking
    total_batches = len(processed_texts) // batch_size + (1 if len(processed_texts) % batch_size > 0 else 0)
    logger.info(f"Processing {len(processed_texts)} texts in {total_batches} batches of size {batch_size}")
    
    # Initialize progress tracker for concise console output
    progress = ProgressTracker(total_batches, "Batch processing")
    processed_count = 0
    
    for batch_num, batch in enumerate(chunk_texts(processed_texts, batch_size)):
        
        batch_start_idx = batch_num * batch_size
        success = False
        retry_count = 0
        max_retries = 3
        
        while not success and retry_count < max_retries:
            try:
                # Log detailed batch info to file, show concise progress on console
                logger.debug(f"Processing batch {batch_num + 1}/{total_batches} "
                           f"(texts {batch_start_idx + 1}-{batch_start_idx + len(batch)}) "
                           f"- Progress: {processed_count}/{len(processed_texts)} "
                           f"({processed_count/len(processed_texts)*100:.1f}%)")
                
                response = client.embeddings.create(
                    model=model,
                    input=batch
                )
                
                # Extract embeddings from response
                batch_embeddings = [item.embedding for item in response.data]
                
                # Validate embeddings
                validation_warnings = 0
                for i, embedding in enumerate(batch_embeddings):
                    if not isinstance(embedding, list) or len(embedding) == 0:
                        logger.warning(f"Invalid embedding format for text {batch_start_idx + i + 1}")
                        batch_embeddings[i] = None
                        validation_warnings += 1
                    elif len(embedding) != 1536:
                        logger.warning(f"Unexpected embedding dimension {len(embedding)} for text {batch_start_idx + i + 1}, expected 1536")
                        validation_warnings += 1
                    else:
                        # Ensure all values are floats
                        try:
                            batch_embeddings[i] = [float(x) for x in embedding]
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error converting embedding to floats for text {batch_start_idx + i + 1}: {e}")
                            batch_embeddings[i] = None
                            validation_warnings += 1
                
                # Store embeddings in correct positions
                for i, embedding in enumerate(batch_embeddings):
                    original_idx = text_indices[batch_start_idx + i]
                    embeddings[original_idx] = embedding
                
                processed_count += len(batch)
                success = True
                
                # Log batch completion
                logger.debug(f"Batch {batch_num + 1} completed successfully" + 
                           (f" ({validation_warnings} warnings)" if validation_warnings > 0 else ""))
                
                # Update progress tracker
                progress.update()
                
            except Exception as e:
                retry_count += 1
                if "rate limit" in str(e).lower():
                    wait_time = min(60 * retry_count, 300)  # Wait 60s, 120s, 180s max 300s
                    logger.warning(f"Rate limit hit. Waiting {wait_time} seconds before retry {retry_count}/{max_retries}")
                    print(f"⏳ Rate limit hit. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    wait_time = 10 * retry_count
                    logger.error(f"Error in batch {batch_num + 1}: {e}")
                    logger.info(f"Retrying in {wait_time} seconds... (attempt {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                
                if retry_count >= max_retries:
                    logger.error(f"Failed to process batch {batch_num + 1} after {max_retries} attempts")
                    # Mark this batch as failed (embeddings already initialized as None)
                    processed_count += len(batch)
                    progress.update()
                    success = True  # Move on to next batch
    
    # Count successful embeddings
    successful_embeddings = sum(1 for emb in embeddings if emb is not None)
    progress.complete(successful_embeddings)
    
    return embeddings
