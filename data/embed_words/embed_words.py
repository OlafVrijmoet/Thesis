
# word embedders
from data.embed_words.gensim_embedding.gensim_embedding import gensim_embedding

# services
from services.get_dfs import get_dfs

# contants
from data.splits.constants import SPLITS

def embed_words():

    gensim_embedding()
