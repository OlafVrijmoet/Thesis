
import numpy as np

def embed_text_gensim(params):

    try:
        vector = params.model.get_vector(params.word, norm=True)
    except:
        vector = np.zeros((params.model.vector_size,), dtype=np.float32)
    
    return vector
