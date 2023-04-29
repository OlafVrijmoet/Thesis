
class Process_Stages:

    def __init__(self, lower, only_text, strip_extra_whitespace, spelling_check) -> None:
        self.lower = lower
        self.only_text = only_text
        self.strip_extra_whitespace = strip_extra_whitespace
        self.spelling_check = spelling_check
