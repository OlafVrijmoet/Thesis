
class Measurement_Settings:

    def __init__(self, print_regression: bool, print_classification: bool, save_performance: bool) -> None:
        
        self.print_regression = print_regression
        self.print_classification = print_classification
        self.save_performance = save_performance

    def __getitem__(self, key):
        return getattr(self, key)
