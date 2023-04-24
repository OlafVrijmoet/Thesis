
# libaries
import pandas as pd
import gensim.downloader as gensim_api

# services
from word_embedding.models.services.gensim.embed_text_gensim import embed_text_gensim

# classes
from word_embedding.classes.Embed_Words import Embed_Words

# print
from services.printing.print_chapter import print_sub_chapter_start, print_sub_chapter_end

# constants
from constants import *
from word_embedding.constants import *

def embed_words():

    print("data/lemmitized_data/datasets")
    # LEMMITIZED_DATASETS

    print_sub_chapter_start(CHAPTER_1)

    # neural
    nerual = Embed_Words(
        name_df="neural_course",
        name_model="fasttext",

        df = pd.read_csv("./data/processed/data/lemmitized_data/domain/neural_networks.csv"),

        model=gensim_api.load("fasttext-wiki-news-subwords-300"),

        embed_word=embed_text_gensim,

        save_path=f"{WORD_EMBEDDING_DATA}/data/lemmitized_data/datasets",

    )

    print_sub_chapter_end(CHAPTER_1)
    print_sub_chapter_start(CHAPTER_2)

    nerual.embed_df()
    nerual.save()

    print_sub_chapter_end(CHAPTER_2)
