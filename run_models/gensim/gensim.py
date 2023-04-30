
# libaries
import os
import gensim.downloader as gensim_api
import pd

# classes
from word_embedding.classes.Embed_Words import Embed_Words
from word_embedding.models.services.gensim.embed_text_gensim import embed_text_gensim
from classes.Datasets import Datasets
from classes.Process_Stages import Process_Stages
from word_embedding.models.classes.EmbeddingModel import EmbeddingModel
from run_models.gensim.classes.Dataset_Gensim import Dataset_Gensim

# services
from run_models.gensim.services.gensim_services import gensim_download, gensim_save, gensim_load

# print
from services.printing.print_chapter import print_sub_chapter_start, print_sub_chapter_end

# constants
from data.splits.constants import *
from word_embedding.models.constants import *
from run_models.gensim.constants import GENSIM_DATA

def gensim():

    datasets = {}
    df_names = []

    # get all file names in a dir
    for file in os.listdir(SPLITS):
        filename, file_extension = os.path.splitext(file)
        df_names.append(filename)

    # move datasets into DatasetClass
    for df_name in df_names:
        print(df_name)
        datasets[df_name] = Dataset_Gensim(
            df_name=df_name,
            model_name="gensim",

            language="english",

            process_stages=Process_Stages(
                lower=True, 
                only_text=True, 
                strip_extra_whitespace=True, 
                spelling_check=True,
                strip_punctuation=True,
                gensim_remove_stop_words=True,
                gensim_tokenization=True,
                gensim_lemmatize=True,
            )
        )

        # get dataset
        datasets[df_name].get_dataset()

        # process dataset based on process_stages
        datasets[df_name].process_dataset()
        
        # save dataset based on df_name at and stage
        datasets[df_name].save()

    # gensim_dataset = Datasets(
    #     force_re_run = False,

    #     base_dir=SPLITS,
    #     model_name="gensim",

    #     progress_stages=Process_Stages(
    #         lower=True, 
    #         only_text=True, 
    #         strip_extra_whitespace=True, 
    #         spelling_check=True,
    #         strip_punctuation=True,
    #         gensim_remove_stop_words=True,
    #         gensim_tokenization=True,
    #         gensim_lemmatize=True,
    #     ),

    #     language="english"
    # )

    # gensim_dataset.get_datasets()

    # models
    models = {
        "fasttext": EmbeddingModel(
            model_name="fasttext",
            download_link="fasttext-wiki-news-subwords-300",
            download_func=gensim_download,
            save_func=gensim_save,
            load_func=gensim_load
        ),

        "glove": EmbeddingModel(
            model_name="glove",
            download_link="glove-wiki-gigaword-300",
            download_func=gensim_download,
            save_func=gensim_save,
            load_func=gensim_load
        ),

        "conceptnet": EmbeddingModel(
            model_name="conceptnet",
            download_link="conceptnet-numberbatch-17-06-300",
            download_func=gensim_download,
            save_func=gensim_save,
            load_func=gensim_load
        )

    }

    # loop through embedding models
    for model in models:

        # loop through stemmed datasets
        for root, dirs, files in os.walk(GENSIM_DATA):

            for file in files:
                if file.endswith(".csv"):

                    file_path = os.path.join(root, file)
                    file_name, _ = os.path.splitext(file)

                    print(f"Processing CSV file: {file_name}")

                    # load data into class for embedding
                    embed_df = Embed_Words(

                        name_df=file_name,
                        name_model=model.model_name,

                        df = pd.read_csv(file_path),

                        model=model,

                        embed_word=embed_text_gensim,

                        save_path=None,

                    )
                    
                    print_sub_chapter_start(f"Embed {file_name}")

                    # embed the data inside the dataset
                    embed_df.embed_df()

                    print_sub_chapter_end(f"Embed {file_name}")

                    # train, test and validate model
