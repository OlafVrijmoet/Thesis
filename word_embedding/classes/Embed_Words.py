
# classes
from word_embedding.classes.Embed_Word_Params import Embed_Word_Params

# services
from services.save import save

# constants
from constants_dir.column_constants import *

class Embed_Words:

    def __init__(self, name_df, name_model, df, model, embed_word, save_path):
        
        self.name_df = name_df
        self.name_model = name_model
        self.df = df

        self.model = model

        # custom function for model that embeds word
        self.embed_word = embed_word

        self.save_path = save_path

    def embed_df(self):

        embedded_reference_answers = []
        embedded_student_answers = []
        
        for index, row in self.df.iterrows():
            
            # embed reference answer
            embedded_ref = self.embed_text(row[REFERENCE_ANSWER])
            embedded_reference_answers.append(embedded_ref)

            # embed student answer
            embedded_ans = self.embed_text(row[STUDENT_ANSWER])
            embedded_student_answers.append(embedded_ans)

    def embed_text(self, text):

        embedded_text = []

        for word in text:

            # embed word using own model and given embed function
            embedded_word = self.embed_word(Embed_Word_Params(self.model, word))

            # add embeded word to embedded text
            embedded_text.append(embedded_word)
        
        return embedded_text
    
    def save(self):

        # save raw
        save(
            dir=self.save_path,
            file_name=self.name,
            df=self.df
        )
