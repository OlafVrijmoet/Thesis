
from classes.Process_Stages import Process_Stages

class Process_Stages_Gensim(Process_Stages):

    def __init__(self, lower, only_text, strip_extra_whitespace, spelling_check, strip_punctuation, gensim_remove_stop_words, gensim_tokenization, gensim_lemmatize) -> None:
        super().__init__(lower, only_text, strip_extra_whitespace, spelling_check, strip_punctuation)

        # Gensim
        self.gensim_remove_stop_words = gensim_remove_stop_words
        self.gensim_tokenization = gensim_tokenization
        self.gensim_lemmatize = gensim_lemmatize

    