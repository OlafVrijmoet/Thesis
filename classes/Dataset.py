
# libaries
import os
import pandas as pd

# libaries
import nltk
import string
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker

from tqdm import tqdm

# services
from services.import_csvs_from_dir import import_csvs_from_dir

# classes
from classes.Process_Stages import Process_Stages

# constants
from data.splits.constants import *

class Dataset:

    def __init__(self, df_name, model_name) -> None:

        self.df_name = df_name
        self.model_name = model_name

        self.basic_processed_df = None

        self.process_stages = Process_Stages(
            lower=True, 
            only_text=True, 
            strip_extra_whitespace=True, 
            spelling_check=True,
            strip_punctuation=True
        )

        self.spell = SpellChecker()

    def get_dataset(self):

        # check if there is already a dataset that has been processed before
        processed_dataset_exists = self.processed_dataset()

        if processed_dataset_exists == False:
            # fetch base dataset from data/splits/self.df_name
            self.basic_processed_df = pd.read_csv(f"{SPLITS}/{self.df_name}.csv")

            # replace Nan answers with emty string
            self.basic_processed_df[["student_answer", "reference_answer", "question"]] = self.basic_processed_df[["student_answer", "reference_answer", "question"]].fillna('')

        else:
            # make sure no basic process stuff is done
            self.process_stages.lower = False
            self.process_stages.only_text = False
            self.process_stages.strip_extra_whitespace = False
            self.process_stages.spelling_check = False

    def processed_dataset(self) -> bool:

        # check if basic processing already done, located at data_saved/basic_processed/df_name
        if os.path.exists(f"data_saved/basic_processed/{self.df_name}.csv"):
            # fetch data and save it to self.basic_processed_df
            self.basic_processed_df = pd.read_csv(f"data_saved/basic_processed/{self.df_name}.csv")
            return True
        else:
            return False

    def process_dataset(self):

        # ensure there is soemthing to be updated in the df
        if self.process_stages.all_true() == True:

            # itterate rows of basic_processed_df
            for index, row in tqdm(self.basic_processed_df.iterrows(), total=self.basic_processed_df.shape[0]):

                # process row
                processed_row = self.process_row(row)
                self.basic_processed_df.loc[index] = processed_row

    def process_row(self, row):

        # lower
        if self.process_stages.lower == True:
            row["student_answer"] = row["student_answer"].lower()
            row["reference_answer"] = row["reference_answer"].lower()

        # remove non chars
        if self.process_stages.only_text == True:
            row["student_answer"] = self.keep_only_text(row["student_answer"])
            row["reference_answer"] = self.keep_only_text(row["reference_answer"])

        # remove extra whitespace
        if self.process_stages.strip_extra_whitespace == True:
            row["student_answer"] = self.strip_extra_whitespace(row["student_answer"])
            row["reference_answer"] = self.strip_extra_whitespace(row["reference_answer"])

        # remove punctuation
        if self.process_stages.strip_punctuation == True:
            row["student_answer"] = self.strip_punctuation(row["student_answer"])
            row["reference_answer"] = self.strip_punctuation(row["reference_answer"])

        # spelling check
        if self.process_stages.spelling_check == True:
            row["student_answer"] = self.correctSpelling(row["student_answer"])
            row["reference_answer"] = self.correctSpelling(row["reference_answer"])

        return row

    def save(self):

        # save basic_processed_df at data_saved/basic_processed/df_name, create dir if it doesn't exist yet
        if not os.path.exists("data_saved/basic_processed"):
            os.makedirs("data_saved/basic_processed")
        self.basic_processed_df.to_csv(f"data_saved/basic_processed/{self.df_name}.csv", index=False)

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
