def chunk_document(text, token_limit=500):
    words = text.split()
    words_per_chunk = int(token_limit * 0.75) 
    
    chunks = []
    for i in range(0, len(words), words_per_chunk):
        chunk_words = words[i : i + words_per_chunk]
        chunk_text = " ".join(chunk_words)
        chunks.append(chunk_text)
        
    return chunks