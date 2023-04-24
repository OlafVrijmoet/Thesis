
def embed_text_gensim(params):

    return params.model.get_vector(params.word, norm=True)
