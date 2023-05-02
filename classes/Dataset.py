
# libaries
import os
import pandas as pd

# libaries
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from nltk.stem.snowball import SnowballStemmer
from spellchecker import SpellChecker
import string

from tqdm import tqdm

# services
from services.import_csvs_from_dir import import_csvs_from_dir
from services.save import save
from services.get_df import get_df

# classes
from classes.Process_Stages import Process_Stages

# constants
from data.splits.constants import *
from run_models.gensim.constants import GENSIM_DATA
from constants_dir.path_constants import BASIC_PROCCESSED, DATA_STAGES

# dataset porcessing
class Dataset:

    def __init__(self, df_name, model_name, language, process_stages, force_basic_processing=False) -> None:

        self.df_name = df_name
        self.model_name = model_name

        self.df = {
            "standardized_splits": None,
            "basic_processed": None
        }
        self.latest_already_processed_phase = "standardized_splits" # this indicates the df that should be used for itteration

        self.language = language

        self.process_stages = process_stages

        self.spell = SpellChecker()
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    # get dataset, process dataset, save dataset
    def run_all(self):

        self.get_dataset()
        self.process_dataset()
        self.save()

    def get_dataset(self):

        for key, df in self.df.items():
            
            # fetch standardized_splits from special dir
            if key == "standardized_splits":
                self.fetch_dataset_and_replace_null(key=key, dir=f"{SPLITS}/{self.df_name}.csv")

            else:
                
                # check if basic processing already done, located at data_saved/basic_processed/df_name
                df_found, df = get_df(
                    dir=f"{DATA_STAGES}/{key}", 
                    file_name=self.df_name
                )

                print(f"df found: {df_found}, name: {key}")

                if df_found:

                    self.df[key] = df
                    self.replace_non_with_string(key)

                    self.latest_already_processed_phase = key

                    self.process_stages[key] = False

                else:
                    # make sure this stage stuff already done indicated 
                    self.process_stages[key] = True

        
        # use the latest dataset and copy it to all none valued in self.df
        for key, processed_row in self.df.items():

            if self.df[key] is None:

                self.df[key] = self.df[self.latest_already_processed_phase].copy()

    def fetch_dataset_and_replace_null(self, key, dir):
        # fetch base dataset from data/splits/self.df_name
        self.df[key] = pd.read_csv(dir)

        self.replace_non_with_string(key)

    def replace_non_with_string(self, key):
        
        # replace Nan answers with emty string
        self.df[key][["student_answer", "reference_answer", "question"]] = self.df[key][["student_answer", "reference_answer", "question"]].fillna('')

    def process_dataset(self):

        # ensure there is something to be updated in the df
        if self.process_stages.any_process_stages_true() == True:
            
            print(f"basic processing starting for {self.df_name}")

            # itterate rows of df
            for index, row in tqdm(self.df[self.latest_already_processed_phase].iterrows(), total=self.df[self.latest_already_processed_phase].shape[0]):

                # process row
                processed_row_dict = self.process_row(row)
                for key, processed_row in processed_row_dict.items():
                    self.df[key].loc[index] = processed_row

        else:
            print(f"basic processing already done for {self.df_name}")

    def process_row(self, row):

        row_dict = {
            "basic_processed": row
        }

        # lower
        if self.process_stages.basic_processed == True:
            row_dict["basic_processed"]["student_answer"] = row_dict["basic_processed"]["student_answer"].lower()
            row_dict["basic_processed"]["reference_answer"] = row_dict["basic_processed"]["reference_answer"].lower()

        # remove non chars
        if self.process_stages.basic_processed == True:
            row_dict["basic_processed"]["student_answer"] = self.keep_only_text(row_dict["basic_processed"]["student_answer"])
            row_dict["basic_processed"]["reference_answer"] = self.keep_only_text(row_dict["basic_processed"]["reference_answer"])

        # remove extra whitespace
        if self.process_stages.basic_processed == True:
            row_dict["basic_processed"]["student_answer"] = self.strip_extra_whitespace(row_dict["basic_processed"]["student_answer"])
            row_dict["basic_processed"]["reference_answer"] = self.strip_extra_whitespace(row_dict["basic_processed"]["reference_answer"])

        # remove punctuation
        if self.process_stages.basic_processed == True:
            row_dict["basic_processed"]["student_answer"] = self.strip_punctuation(row_dict["basic_processed"]["student_answer"])
            row_dict["basic_processed"]["reference_answer"] = self.strip_punctuation(row_dict["basic_processed"]["reference_answer"])

        # spelling check
        if self.process_stages.basic_processed == True:
            row_dict["basic_processed"]["student_answer"] = self.correctSpelling(row_dict["basic_processed"]["student_answer"])
            row_dict["basic_processed"]["reference_answer"] = self.correctSpelling(row_dict["basic_processed"]["reference_answer"])

        return row_dict

    def keep_only_text(self, text: str) -> str:

        # define the regular expression pattern
        pattern = r'[^\w\s]'

        return text.replace(pattern, '')

    def strip_extra_whitespace(self, s: str) -> str:
        return " ".join(s.split())

    def strip_punctuation(self, text: str) -> str:

        # remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        return text
    
    def correctSpelling(self, text):

        if not text:
            return ""

        words = text.split()

        corrected_words = [self.spell.correction(word) if word != self.spell.correction(word) and self.spell.correction(word) is not None else word for word in words]
        corrected_sentence = " ".join(corrected_words)

        # Initialize the spell checker
        return corrected_sentence

    # save basic processed
    def save(self):
        
        for key, df in self.df.items():

            # skip the base df
            if key == "standardized_splits":
                continue

            # check if any new basic processing is done
            if self.process_stages[key] == True:
                print(f"saving new {key} phase for: {self.df_name}")
                save(
                    dir=f"{DATA_STAGES}/{key}",
                    file_name=self.df_name,
                    df=df
                )
            else:
                print(f"no saving needed because basic processing already done on {self.df_name}")
