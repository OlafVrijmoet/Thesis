
import numpy as np

# classes
from classes.Grading_Model import Grading_Model

# constants
from performance_tracking.constants import ALL, TRAIN, TEST, VALIDATION

class Openai_Grading(Grading_Model):

    def __init__(self,
        # parent
        model, dataset, measurement_settings,

        # child
        x_column, y_column
    ):
        """
        Initialize the Regression_Grading class.

        Parameters:
        - model: Measurement_Settings
          The model Measurement_Settings and contains information already known to create an identifiable experiment in the measurements.
        - dataset: Dataset
          The dataset object.
        - trained: bool
            ensures that the model is not trained multiple times
        """
        super().__init__(model, dataset, measurement_settings, y_column)

        self.x_column=x_column
        
        # self.trained = False
