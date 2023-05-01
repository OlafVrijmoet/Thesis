
import os
from tqdm import tqdm

import pandas as pd

from scipy.spatial.distance import cosine
import nltk

from ast import literal_eval

# services
from services.save import save
from services.get_df import get_df

# parant classes
from classes.Dataset import Dataset

# constants
from run_models.gensim.constants import GENSIM_DATA
from constants_dir.column_constants import *
from constants_dir.path_constants import DATA_STAGES
from constants_dir.path_constants import BASIC_PROCCESSED, DATA_STAGES, GENSIM_PROCESSED

class Dataset_Gensim(Dataset):

    def __init__(self, df_name, model_name, language, process_stages, force_Gensim_run=False) -> None:
        super().__init__(df_name, model_name, language, process_stages)

        self.former_stage_dir = f"{BASIC_PROCCESSED}"
        self.next_stage_dir = f"{DATA_STAGES}/{model_name}"
        self.force_Gensim_run = force_Gensim_run

    def get_dataset(self):
        # make sure that basic processing is done, get the already basic processed df from dir if done, else do the basic processing
        super().get_dataset()

        # by default gensim is not already processed
        gensim_processed_dataset_exists = False
        # check if the Gensim processing is done, get the already basic processed df from dir if done, else do the gensim processing
        if self.force_Gensim_run == False:
            gensim_processed_dataset_exists = self.gensim_processed_datasets()            

        if gensim_processed_dataset_exists == False:
            # fetch datasets from basic processed
            self.df = pd.read_csv(f"{BASIC_PROCCESSED}/{self.df_name}.csv")

            # replace Nan answers with emty string
            self.df[["student_answer", "reference_answer", "question"]] = self.df[["student_answer", "reference_answer", "question"]].fillna('')

        # if force run again keep them true
        elif not self.force_Gensim_run == True:
            # no need to do these opperations again
            self.process_stages.gensim_remove_stop_words = False
            self.process_stages.gensim_tokenization = False
            self.process_stages.gensim_lemmatize = False

    def gensim_processed_datasets(self) -> bool:
        if self.process_stages.gensim_remove_stop_words == True:
            print("downloading nltk stopwords")
            nltk.download('stopwords')

        if self.process_stages.gensim_lemmatize == True:
            print("downloading nltk wordnet")
            nltk.download('wordnet')

        # check if gensim processing already done, located at data_saved/basic_processed/df_name
        df_found, df = get_df(
            dir=GENSIM_PROCESSED, 
            file_name=self.df_name
        )

        if df_found == True:
            self.df = df
            return True
        
        else:
            return False

    def process_row(self, row):
        # include all processing from the apprent class function
        super().process_row(row)

        if self.process_stages.gensim_tokenization == True:
            row["student_answer"] = nltk.word_tokenize(row["student_answer"])
            row["reference_answer"] = nltk.word_tokenize(row["reference_answer"])

        if self.process_stages.gensim_remove_stop_words == True:
            row["student_answer"] = self.remove_stop_words(row["student_answer"])
            row["reference_answer"] = self.remove_stop_words(row["reference_answer"])

        if self.process_stages.gensim_lemmatize == True:
            row["student_answer"] = self.lemmatize_tokens(row["student_answer"])
            row["reference_answer"] = self.lemmatize_tokens(row["reference_answer"])

        # Remove empty tokens that may have been created during preprocessing
        # if self.process_stages.gensim_tokenization == True:
        #     tokens = [token for token in tokens if token]

        return row

    def remove_stop_words(self, tokens):

        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):

        return [self.lemmatizer.lemmatize(token) for token in tokens]

    # def save(self):

    #     if self.process_stages.any_process_stages_true() == True or self.force_Gensim_run == True:
    #         print(f"saving gensim processing: {self.df_name}")
    #         save(
    #             dir=GENSIM_PROCESSED,
    #             file_name=self.df_name,
    #             df=self.df
    #         )
    #     else:
    #         print(f"no saving needed because gensim processing already done on {self.df_name}")

        # path = f"{DATA_STAGES}/{self.model_name}/{self.df_name}.csv"
        # if self.process_stages.gensim_tokenization == True:
        #     path = GENSIM_DATA
        # # save basic_processed_df at data_saved/basic_processed/df_name, create dir if it doesn't exist yet
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # self.basic_processed_df.to_csv(f"{path}/{self.df_name}.csv", index=False)
