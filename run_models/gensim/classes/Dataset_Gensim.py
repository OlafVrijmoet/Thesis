
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

        self.df = {
            "standardized_splits": None,
            "basic_processed": None,
            "gensim": None
        }

    def get_dataset(self):
        # make sure that basic processing is done, get the already basic processed df from dir if done, else do the basic processing
        super().get_dataset()

        # things needed to do the gensim pre-processing
        print("downloading nltk stopwords")
        nltk.download('stopwords')

        print("downloading nltk wordnet")
        nltk.download('wordnet')

    def process_row(self, row):
        # include all processing from the apprent class function
        row_dict = super().process_row(row)

        # add gensim version of row
        row_dict[self.model_name] = row_dict["basic_processed"].copy()

        if self.process_stages.gensim == True:
            row_dict[self.model_name]["student_answer"] = nltk.word_tokenize(row_dict[self.model_name]["student_answer"])
            row_dict[self.model_name]["reference_answer"] = nltk.word_tokenize(row_dict[self.model_name]["reference_answer"])

        if self.process_stages.gensim == True:
            row_dict[self.model_name]["student_answer"] = self.remove_stop_words(row_dict[self.model_name]["student_answer"])
            row_dict[self.model_name]["reference_answer"] = self.remove_stop_words(row_dict[self.model_name]["reference_answer"])

        if self.process_stages.gensim == True:
            row_dict[self.model_name]["student_answer"] = self.lemmatize_tokens(row_dict[self.model_name]["student_answer"])
            row_dict[self.model_name]["reference_answer"] = self.lemmatize_tokens(row_dict[self.model_name]["reference_answer"])

        # Remove empty tokens that may have been created during preprocessing
        # if self.process_stages.gensim_tokenization == True:
        #     tokens = [token for token in tokens if token]

        return row_dict

    def remove_stop_words(self, tokens):

        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):

        return [self.lemmatizer.lemmatize(token) for token in tokens]
