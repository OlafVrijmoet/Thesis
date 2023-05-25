
import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

# classes
from performance_tracking.classes.Performance_Row import Performance_Row

# constants
from performance_tracking.constants import ALL, TRAIN, TEST, VALIDATION

class Grading_Model:

    def __init__(self, model, dataset, measurement_settings, y_column):
        """
        Initialize the Grading_Model class.

        Parameters:
        - model: object
            The model object.
        - dataset: Dataset
            The dataset object.
        - experiment_identification: Measurement_Settings
          The model Measurement_Settings and contains information already known to create an identifiable experiment in the measurements.
        """
        self.model = model
        self.dataset = dataset
        self.measurement_settings = measurement_settings
        self.y_column = y_column

    def train(self):
        """
        *** customize ***
        Train the grading model.
        """
        # measure performance
        self.measure_performance(
            dataset_split=TRAIN,
        )

    def test(self):
        """
        *** customize ***
        Test the grading model.
        """

        # measure performance on test datasetsplit
        self.measure_performance(
            dataset_split=TEST,
        )

    def validation(self):
        """
        *** customize ***
        Perform validation for the grading model.
        """

        # measure performance on validation datasetsplit
        self.measure_performance(
            dataset_split=VALIDATION,
        )

    def measure_performance(self, dataset_split):
        """
        Measure the performance of the grading model on a dataset split.

        Parameters:
        - dataset_split: str
          The dataset split to measure performance on.
        """
        # create Performance_Row class
        performance_tracking = Performance_Row(
            
            # id what is being tracked
            dataset_name=self.measurement_settings.dataset_name,
            embedding_seperated=self.measurement_settings.embedding_seperated,
            embedding_model_name = self.measurement_settings.embedding_model_name,
            sentence_embedding_method = self.measurement_settings.sentence_embedding_method,
            feature_engenearing_method = self.measurement_settings.feature_engenearing_method,
            grading_model = self.measurement_settings.grading_model,
            seed_data_split = self.measurement_settings.seed_data_split,

            dataset_split=dataset_split,

            # duplicates handeling
            settings_performance_tracking=self.measurement_settings.settings_performance_tracking
        )

        # print model info settings ask to print performance
        if self.measurement_settings.print_regression == True or self.measurement_settings.print_classification == True:

            performance_tracking.print_experiement_info()

        # make predictions if print_regression, print_classification or save_performance are true
        if self.measurement_settings.print_regression == True or self.measurement_settings.print_classification == True or self.measurement_settings.save_performance == True:

            y_pred = self.make_predictions(dataset_split)

        if self.measurement_settings.print_regression == True or self.measurement_settings.save_performance == True:

            # measure regression accuracy
            performance_tracking = self.mean_squared_error(performance_tracking, dataset_split, y_pred)

            # print accuracy regression
            if self.measurement_settings.print_regression == True:

                performance_tracking.print_regression_preformance()

        if self.measurement_settings.print_classification == True or self.measurement_settings.save_performance == True:
            
            # measure classification performance
            performance_tracking = self.classification_performance(performance_tracking, dataset_split, y_pred)

            if self.measurement_settings.print_classification == True:
                
                performance_tracking.print_classification_performance()

        if self.measurement_settings.save_performance == True:

            # run saving
            performance_tracking.save()

    def make_predictions(self):
        """
        *** customize ***
        Make predictions using the grading model.
        """

        raise ValueError("No custome make_predictions fuction defined in the child class")

    def mean_squared_error(self, performance_tracking, dataset_split, y_pred):
        """
        Calculate the mean squared error (MSE) between true and predicted values.

        Parameters:
        - performance_tracking: Performance_Row
            the row data for performance tracking
        - y_true: array-like
          The true values.
        - y_pred: array-like
          The predicted values.
        - avg_max_points: float
            average max points values in dataset

        Returns:
        - float
          The calculated mean squared error.
        """

        # get ground truth
        y_true = self.dataset[dataset_split][self.y_column]

        # get average max points
        avg_max_points = self.dataset[dataset_split]["max_points"].mean()

        # Pearson's correlation
        correlation, p_value = pearsonr(y_true, y_pred)
        performance_tracking['pears_correlation'] = correlation
        performance_tracking['p_value'] = p_value

        #!!!!!! sqrt means no mead for ** 2 of avg_max_points !!!!!!!!
        # FOR MSE: rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * (avg_max_points ** 2)

        # the error (distance between normalized pred and normalized actual value) is squared for rmse, so the avg_max_points also has to be squared to get the correct non-normalized rmse
        rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * avg_max_points

        # save rmse for experiement
        performance_tracking['rmse'] = rmse

        # i think necissary
        return performance_tracking

    def classification_performance(self, performance_tracking, dataset_split, y_pred):
        """
        Evaluate the classification performance based on true and predicted labels.

        Parameters:
        - performance_tracking: Performance_Row
            the row data for performance tracking
        - y_pred: array-like
          The predicted labels.

        Returns:
        - dict
          A dictionary containing various classification performance metrics.
        """

        # add predictions to the df as a column
        self.dataset[dataset_split]["y_pred"] = y_pred

        # scale prediction from normalized value back to the original points
        self.dataset[dataset_split]["pred_points"] = self.dataset[dataset_split]["y_pred"] * self.dataset[dataset_split]["max_points"]

        # round to closest round number and convert from float to int
        self.dataset[dataset_split]["pred_points"] = self.dataset[dataset_split]["pred_points"].round()
        self.dataset[dataset_split]["pred_points"] = self.dataset[dataset_split]["pred_points"].astype(int)

        # Calculate accuracy
        accuracy = accuracy_score(self.dataset[dataset_split]["assigned_points"], self.dataset[dataset_split]["pred_points"])

        # Calculate precision, recall, and F1-score
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(self.dataset[dataset_split]["assigned_points"], self.dataset[dataset_split]["pred_points"], average='macro')
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(self.dataset[dataset_split]["assigned_points"], self.dataset[dataset_split]["pred_points"], average='micro')
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(self.dataset[dataset_split]["assigned_points"], self.dataset[dataset_split]["pred_points"], average='weighted')

        performance_tracking['accuracy'] = accuracy
        performance_tracking['precision_macro'] = precision_macro
        performance_tracking['recall_macro'] = recall_macro
        performance_tracking['f1_macro'] = f1_macro
        performance_tracking['precision_micro'] = precision_micro
        performance_tracking['recall_micro'] = recall_micro
        performance_tracking['f1_micro'] = f1_micro
        performance_tracking['precision_weighted'] = precision_weighted
        performance_tracking['recall_weighted'] = recall_weighted
        performance_tracking['f1_weighted'] = f1_weighted

        return performance_tracking
