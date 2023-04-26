
# libaries
import os
import pandas as pd
import gensim.downloader as gensim_api

# services
from word_embedding.models.services.gensim.load_gensim_model import gensim_download, gensim_save, gensim_load
from word_embedding.models.services.gensim.embed_text_gensim import embed_text_gensim

# classes
from word_embedding.classes.Embed_Words import Embed_Words
from word_embedding.models.classes.EmbeddingModel import EmbeddingModel

# print
from services.printing.print_chapter import print_sub_chapter_start, print_sub_chapter_end

# constants
from constants_dir.path_constants import LEMMITIZED_DATASETS
from word_embedding.constants import WORD_EMBEDDING, WORD_EMBEDDING_DATA, CHAPTER_1, CHAPTER_2

def embed_words():

    print_sub_chapter_start(CHAPTER_1)

    fasttext = EmbeddingModel(
        model_name="fasttext",
        download_link="fasttext-wiki-news-subwords-300",
        download_func=gensim_download,
        save_func=gensim_save,
        load_func=gensim_load
    )

    glove = EmbeddingModel(
        model_name="glove",
        download_link="glove-wiki-gigaword-300",
        download_func=gensim_download,
        save_func=gensim_save,
        load_func=gensim_load
    )

    word2vec = EmbeddingModel(
        model_name="word2vec",
        download_link="word2vec-google-news-300",
        download_func=gensim_download,
        save_func=gensim_save,
        load_func=gensim_load
    )

    conceptnet = EmbeddingModel(
        model_name="conceptnet",
        download_link="conceptnet-numberbatch-17-06-300",
        download_func=gensim_download,
        save_func=gensim_save,
        load_func=gensim_load
    )

    fasttext.load_model()
    glove.load_model()
    word2vec.load_model()
    conceptnet.load_model()

    # neural
    nerual = Embed_Words(
        name_df="neural_course",
        name_model="fasttext",

        df = pd.read_csv("./data/processed/data/lemmitized_data/domain/neural_networks.csv"),

        model=fasttext,

        embed_word=embed_text_gensim,

        path_save_model=f"",
        save_path=f"{WORD_EMBEDDING}/{LEMMITIZED_DATASETS}",

    )

    print_sub_chapter_end(CHAPTER_1)
    print_sub_chapter_start(CHAPTER_2)

    nerual.embed_df()
    nerual.save()

    print_sub_chapter_end(CHAPTER_2)
