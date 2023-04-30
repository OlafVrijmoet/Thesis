
import os
from tqdm import tqdm

from scipy.spatial.distance import cosine
import nltk

from ast import literal_eval

# parant classes
from classes.Dataset import Dataset

# constants
from run_models.gensim.constants import GENSIM_DATA
from constants_dir.column_constants import *

class Dataset_Gensim(Dataset):

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

    def embed_df(self):

        embedded_reference_answers = []
        embedded_student_answers = []
        
        for index, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):

            # embed reference answer
            embedded_ref = self.embed_text(literal_eval(row[REFERENCE_ANSWER]))
            embedded_reference_answers.append(embedded_ref)

            # embed student answer
            embedded_ans = self.embed_text(literal_eval(row[STUDENT_ANSWER]))
            embedded_student_answers.append(embedded_ans)

        self.df["reference_answer_embedding"] = embedded_reference_answers
        self.df["student_answer_embedding"] = embedded_student_answers

    def cosine_similarity(self, vec1, vec2):

        return 1 - cosine(vec1, vec2)
    
    def add_cosine_similarity_column(self):
        self.df['cosine_similarity'] = self.df.apply(lambda row: self.cosine_similarity(row['student_answer_embedding'], row['reference_answer_embedding']), axis=1)

    def save(self):
        path = "data_saved/basic_processed"
        if self.process_stages.gensim_tokenization == True:
            path = GENSIM_DATA
        # save basic_processed_df at data_saved/basic_processed/df_name, create dir if it doesn't exist yet
        if not os.path.exists(path):
            os.makedirs(path)
        self.basic_processed_df.to_csv(f"{path}/{self.df_name}.csv", index=False)
