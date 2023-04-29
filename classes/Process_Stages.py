
class Process_Stages:

    def __init__(self, lower, only_text, strip_extra_whitespace, spelling_check, strip_punctuation) -> None:
        self.lower = lower
        self.only_text = only_text
        self.strip_extra_whitespace = strip_extra_whitespace
        self.spelling_check = spelling_check
        self.strip_punctuation = strip_punctuation

    def all_true(self):
        return all([self.lower, self.only_text, self.strip_extra_whitespace, self.spelling_check, self.strip_punctuation])
