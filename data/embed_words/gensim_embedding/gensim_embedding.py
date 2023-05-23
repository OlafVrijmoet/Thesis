
# libaries
import os
import gensim.downloader as gensim_api
import pandas as pd

# classes
from classes.Dataset_Settings import Dataset_Settings

# local classes
from data.embed_words.gensim_embedding.classes.Dataset_Gensim import Dataset_Gensim
from data.embed_words.gensim_embedding.classes.Gensim_Embedding_Model import Gensim_Embedding_Model
from data.embed_words.gensim_embedding.classes.Embedding_Model_Gensim import Embedding_Model_Gensim

# local services
from data.embed_words.gensim_embedding.services.Gensim_services import gensim_download, gensim_save, gensim_load

# print
from services.printing.print_chapter import print_sub_chapter_start, print_sub_chapter_end

# constants
from data.splits.constants import SPLITS
from word_embedding.models.constants import *
from run_models.gensim.constants import GENSIM_DATA

# local constants
from data.embed_words.constants import EMBED_WORDS
from data.embed_words.gensim_embedding.constants import GENSIM_MODEL_NAMES

def gensim_embedding():

    datasets = {}
    df_names = []

    # get all file names in a dir
    for file in os.listdir(SPLITS):
        filename, file_extension = os.path.splitext(file)
        df_names.append(filename)

    # move datasets into DatasetClass
    for df_name in df_names:
        print(f"move datasets into DatasetClass: {df_name}")

        datasets[df_name] = Dataset_Gensim(
            df_name=df_name,
            model_name="gensim", # used for dir name inside data_saved

            language="english",

            datasets = {
                "standardized_splits": Dataset_Settings(
                    df=None,
                    may_run_now=False,
                    required=True
                ),
                "basic_processed": Dataset_Settings(
                    df=None,
                    may_run_now=True,
                    required=True
                ),
                "gensim": Dataset_Settings(
                    df=None,
                    may_run_now=True,
                    required=True
                ),
            },

        )

        # get dataset, process dataset, save dataset
        datasets[df_name].run_all()

    # normalize points for each dataset
    for df_name in df_names:
        print(f"normalizing points for dataset: {df_name}")

        df = pd.read_csv(f"data_saved/gensim/{df_name}.csv")

        # round values to howl int numbers
        df["assigned_points"] = df["assigned_points"].round()
        df["assigned_points"] = df["assigned_points"].astype(int)

        # add normalized points
        df["normalized_points"] = df["assigned_points"] / df["max_points"]

        df.to_csv(f"data_saved/gensim/{df_name}.csv", index=False)

    # models
    models = {
        GENSIM_MODEL_NAMES["fasttext"]: Gensim_Embedding_Model(
            model_name=GENSIM_MODEL_NAMES["fasttext"],
            download_link="fasttext-wiki-news-subwords-300",
            download_func=gensim_download,
            save_func=gensim_save,
            load_func=gensim_load
        ),

        GENSIM_MODEL_NAMES["glove"]: Gensim_Embedding_Model(
            model_name=GENSIM_MODEL_NAMES["glove"],
            download_link="glove-wiki-gigaword-300",
            download_func=gensim_download,
            save_func=gensim_save,
            load_func=gensim_load
        ),

        # "word2vec": Gensim_Embedding_Model(
        #     model_name="word2vec",
        #     download_link="word2vec-google-news-300",
        #     download_func=gensim_download,
        #     save_func=gensim_save,
        #     load_func=gensim_load
        # ),

        GENSIM_MODEL_NAMES["conceptnet"]: Gensim_Embedding_Model(
            model_name=GENSIM_MODEL_NAMES["conceptnet"],
            download_link="conceptnet-numberbatch-17-06-300",
            download_func=gensim_download,
            save_func=gensim_save,
            load_func=gensim_load
        )

    }

    # loop through embedding models
    for key, model in models.items():
        
        # download / fetch model
        model.load_model()

        # loop through data sets
        for root, dirs, files in os.walk(f"data_saved/gensim"):

            for df_name in files:

                if df_name.endswith(".csv"):

                    file_path = os.path.join(root, df_name)
                    df_name, _ = os.path.splitext(df_name)

                    print(f"move datasets into DatasetClass: {df_name}")

                    model_name = f"{EMBED_WORDS}_{model.model_name}"

                    dataset = Embedding_Model_Gensim(
                        df_name=df_name,
                        model_name=model_name, # used for dir name inside data_saved

                        language="english",

                        datasets = {
                            "gensim": Dataset_Settings(
                                df=None,
                                may_run_now=False,
                                required=True,
                                force_run=False
                            ),
                            model_name: Dataset_Settings(
                                df=None,
                                may_run_now=True,
                                required=True,
                                parquet=True,
                                name_required_dataset="gensim",
                                force_run=True
                            ),
                        },

                        embedding_model=model

                    )

                    # get dataset, process dataset, save dataset
                    dataset.run_all()
