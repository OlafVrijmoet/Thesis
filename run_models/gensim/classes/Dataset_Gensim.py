
import os
from tqdm import tqdm

import pandas as pd

from scipy.spatial.distance import cosine
import nltk

from ast import literal_eval

# parant classes
from classes.Dataset import Dataset

# constants
from run_models.gensim.constants import GENSIM_DATA
from constants_dir.column_constants import *
from constants_dir.path_constants import DATA_STAGES

class Dataset_Gensim(Dataset):

    # 
    def get_dataset(self):
        # make sure that basic processing is done, get the already basic processed df from dir if done, else do the basic processing
        super().get_dataset()

        # check if the Gensim processing is done, get the already basic processed df from dir if done, else do the gensim processing
        gensim_processed_dataset_exists = self.gensim_processed_datasets()

        if gensim_processed_dataset_exists == False:
            # fetch datasets from basic processed
            self.df = pd.read_csv()

        else:
            # no need to do these opperations again
            self.process_stages.gensim_remove_stop_words = False
            self.process_stages.gensim_tokenization = False
            self.process_stages.gensim_lemmatize = False

    def gensim_processed_datasets(self) -> bool:
        if self.process_stages.gensim_remove_stop_words == True:
            nltk.download('stopwords')

        if self.process_stages.gensim_lemmatize == True:
            nltk.download('wordnet')
        
        # check if gensim processing already done, located at data_saved/basic_processed/df_name
        if os.path.exists(f"{DATA_STAGES}/{self.model_name}/{self.df_name}.csv"):
            # fetch data and save it to self.df
            self.df = pd.read_csv(f"{DATA_STAGES}/{self.model_name}/{self.df_name}.csv")
            return True
        else:
            return False

    def process_row(self, row):
        # include all processing from the apprent class function
        super().process_row()

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
        if self.process_stages.gensim_tokenization == True:
            tokens = [token for token in tokens if token]

    # # adds up the fectors of each word in a sentence to create a sentence embedding
    # def embed_sentence_add(self):

    #     embedded_reference_answers = []
    #     embedded_student_answers = []
        
    #     for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):

    #         # embed reference answer
    #         embedded_ref = self.embed_text(literal_eval(row[REFERENCE_ANSWER]))
    #         embedded_reference_answers.append(embedded_ref)

    #         # embed student answer
    #         embedded_ans = self.embed_text(literal_eval(row[STUDENT_ANSWER]))
    #         embedded_student_answers.append(embedded_ans)

    #     self.df["reference_answer_embedding"] = embedded_reference_answers
    #     self.df["student_answer_embedding"] = embedded_student_answers

    # def embed_text(self, text):

    #     embedded_text = []

    #     for word in text:

    #         # embed word using own model and given embed function
    #         embedded_word = self.embed_word(Embed_Word_Params(self.model.model, word))

    #         # add embeded word to embedded text
    #         embedded_text.append(embedded_word)
        
    #     return embedded_text

    # def cosine_similarity(self, vec1, vec2):

    #     return 1 - cosine(vec1, vec2)
    
    # def add_cosine_similarity_column(self):
    #     self.df['cosine_similarity'] = self.df.apply(lambda row: self.cosine_similarity(row['student_answer_embedding'], row['reference_answer_embedding']), axis=1)

    def save(self):
        path = f"{DATA_STAGES}/{self.model_name}/{self.df_name}.csv"
        if self.process_stages.gensim_tokenization == True:
            path = GENSIM_DATA
        # save basic_processed_df at data_saved/basic_processed/df_name, create dir if it doesn't exist yet
        if not os.path.exists(path):
            os.makedirs(path)
        self.basic_processed_df.to_csv(f"{path}/{self.df_name}.csv", index=False)
