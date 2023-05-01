
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
from constants_dir.path_constants import BASIC_PROCCESSED

# dataset porcessing
class Dataset:

    def __init__(self, df_name, model_name, language, process_stages) -> None:

        self.df_name = df_name
        self.model_name = model_name

        self.df = None

        self.language = language

        self.process_stages = process_stages

        self.spell = SpellChecker()
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

        self.former_stage_dir = f"{SPLITS}"
        self.next_stage_dir = f"{BASIC_PROCCESSED}"

    # get dataset, process dataset, save dataset
    def run_all(self):

        self.get_dataset()
        self.process_dataset()
        self.save()

    def get_dataset(self):

        # check if there is already a dataset that has been processed before
        processed_dataset_exists = self.processed_dataset()

        if processed_dataset_exists == False:
            # fetch base dataset from data/splits/self.df_name
            self.df = pd.read_csv(f"{self.former_stage_dir}/{self.df_name}.csv")

            # replace Nan answers with emty string
            self.df[["student_answer", "reference_answer", "question"]] = self.df[["student_answer", "reference_answer", "question"]].fillna('')

        else:
            # make sure no basic process stuff is done
            self.process_stages.lower = False
            self.process_stages.only_text = False
            self.process_stages.strip_extra_whitespace = False
            self.process_stages.spelling_check = False
            self.process_stages.strip_punctuation = False

    def processed_dataset(self) -> bool:

        # check if basic processing already done, located at data_saved/basic_processed/df_name
        df_found, df = get_df(
            dir=self.next_stage_dir, 
            file_name=self.df_name
        )

        if df_found == True:
            self.df = df
            return True
        
        else:
            return False

        # if self.process_stages.gensim_remove_stop_words == True:
        #     nltk.download('stopwords')

        # if self.process_stages.gensim_lemmatize == True:
        #     nltk.download('wordnet')

        # if os.path.exists(f"{self.next_stage_dir}/{self.df_name}.csv"):
        #     print(f"basic processing exists already for {self.df_name}")
            
        #     # fetch data and save it to self.df
        #     self.df = pd.read_csv(f"{self.next_stage_dir}/{self.df_name}.csv")
        #     return True
        # else:
        #     print(f"basic processing does not already exist for {self.df_name}")
        #     return False

    def process_dataset(self):

        # ensure there is soemthing to be updated in the df
        if self.process_stages.all_basic_processing_true() == True:
            
            print(f"basic processing starting for {self.df_name}")

            # itterate rows of df
            for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):

                # process row
                processed_row = self.process_row(row)
                self.df.loc[index] = processed_row

        else:
            print(f"basic processing already done for {self.df_name}")

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
        
        # # NOW PART OF Dataset_Gensim
        # if self.process_stages.gensim_tokenization == True:
        #     row["student_answer"] = nltk.word_tokenize(row["student_answer"])
        #     row["reference_answer"] = nltk.word_tokenize(row["reference_answer"])

        # # NOW PART OF Dataset_Gensim
        # if self.process_stages.gensim_remove_stop_words == True:
        #     row["student_answer"] = self.remove_stop_words(row["student_answer"])
        #     row["reference_answer"] = self.remove_stop_words(row["reference_answer"])

        # # NOW PART OF Dataset_Gensim
        # if self.process_stages.gensim_lemmatize == True:
        #     row["student_answer"] = self.lemmatize_tokens(row["student_answer"])
        #     row["reference_answer"] = self.lemmatize_tokens(row["reference_answer"])

        # # Remove empty tokens that may have been created during preprocessing
        # if self.process_stages.gensim_tokenization == True:
        #     tokens = [token for token in tokens if token]

        return row

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

    # def remove_stop_words(self, tokens):
    #     tokens = [token for token in tokens if token not in self.stop_words]

    #     return tokens
    
    # def lemmatize_tokens(self, tokens):

    #     tokens = [self.lemmatizer.lemmatize(token) for token in tokens]

    # save basic processed
    def save(self):

        # check if any new basic processing is done
        if self.process_stages.all_basic_processing_true() == True:
            print(f"saving new basis processing: {self.df_name}")
            save(
                dir=self.next_stage_dir,
                file_name=self.df_name,
                df=self.df
            )
        else:
            print(f"no saving needed because new basic processing done on {self.df_name}")
