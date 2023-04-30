
import os
from scipy.spatial.distance import cosine
import nltk

# parant classes
from classes.Dataset import Dataset

# constants
from run_models.gensim.constants import GENSIM_DATA

class Dataset_Gensim(Dataset):

    def process_row(self, row):
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

    def cosine_similarity(vec1, vec2):

        return 1 - cosine(vec1, vec2)
    
    def save(self):
        path = "data_saved/basic_processed"
        if self.process_stages.gensim_tokenization == True:
            path = GENSIM_DATA
        # save basic_processed_df at data_saved/basic_processed/df_name, create dir if it doesn't exist yet
        if not os.path.exists(path):
            os.makedirs(path)
        self.basic_processed_df.to_csv(f"{path}/{self.df_name}.csv", index=False)
