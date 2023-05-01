
class Process_Stages:

    def __init__(self, lower, only_text, strip_extra_whitespace, spelling_check, strip_punctuation, gensim_remove_stop_words, gensim_tokenization, gensim_lemmatize) -> None:
        # part of basic_processed
        self.lower = lower
        self.only_text = only_text
        self.strip_extra_whitespace = strip_extra_whitespace
        self.spelling_check = spelling_check
        self.strip_punctuation = strip_punctuation

        # Gensim
        self.gensim_remove_stop_words = gensim_remove_stop_words
        self.gensim_tokenization = gensim_tokenization
        self.gensim_lemmatize = gensim_lemmatize

    def all_basic_processing_true(self):
        return all([self.lower, self.only_text, self.strip_extra_whitespace, self.spelling_check, self.strip_punctuation])
