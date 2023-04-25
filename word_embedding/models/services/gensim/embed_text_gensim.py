
def embed_text_gensim(params):

    print(params.word)

    return params.model.get_vector(params.word, norm=True)
