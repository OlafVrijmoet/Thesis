
import numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

class Regression_Model():

    def __init__(self, dataset, model, x_column, y_column) -> None:
        
        self.dataset = dataset
        self.model = model

        # columns
        self.x_column = x_column
        self.y_column = y_column

        # performance
        self.rmse_test = None
        self.rmse_validation = None

        self.test_accuracy = None,
        self.test_precision_macro = None, 
        self.test_recall_macro = None, 
        self.test_f1_macro = None,
        self.test_precision_micro = None, 
        self.test_recall_micro = None, 
        self.test_f1_micro = None,
        self.test_precision_weighted = None, 
        self.test_recall_weighted = None, 
        self.test_f1_weighted = None,
        
        self.validation_accuracy = None,
        self.validation_precision_macro = None, 
        self.validation_recall_macro = None, 
        self.validation_f1_macro = None,
        self.validation_precision_micro = None, 
        self.validation_recall_micro = None, 
        self.validation_f1_micro = None,
        self.validation_precision_weighted = None, 
        self.validation_recall_weighted = None, 
        self.validation_f1_weighted = None,

    def train(self):
                
        self.model = self.model().fit(self.dataset["train_df"][self.x_column].values.reshape(-1, 1), self.dataset["train_df"][self.y_column])
    
    def test(self):

        y_pred = self.model.predict(self.dataset["test_df"][self.x_column].values.reshape(-1, 1))

        self.mean_squared_error(test=True, y_ground_truth=self.dataset["test_df"][self.y_column], y_predicted=y_pred)

    def mean_squared_error(self, test: bool, y_ground_truth, y_predicted):

        rmse = np.sqrt(mean_squared_error(y_ground_truth, y_predicted))

        if test == True:

            self.rmse_test = rmse
        
        else:

            self.rmse_validation = rmse

    def model_accuracty(self, test: bool, df_name):

        # make predictions based on df_name (the df split)
        y_pred = self.model.predict(self.dataset[df_name][self.x_column].values.reshape(-1, 1))

        # add predictions to the df as a column
        self.dataset[df_name]["y_pred"] = y_pred

        # scale prediction from normalized value back to the orgininal points
        self.dataset[df_name]["pred_points"] = self.dataset[df_name]["y_pred"] * self.dataset[df_name]["max_points"]

        # round to closest round number and convert from float to int
        self.dataset[df_name]["pred_points"] = self.dataset[df_name]["pred_points"].round()
        self.dataset[df_name]["pred_points"] = self.dataset[df_name]["pred_points"].astype(int)

        if test == True:
            # Calculate accuracy
            self.test_accuracy = accuracy_score(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"])

            # Calculate precision, recall, and F1-score
            self.test_precision_macro, self.test_recall_macro, self.test_f1_macro, _ = precision_recall_fscore_support(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"], average='macro')
            self.test_precision_micro, self.test_recall_micro, self.test_f1_micro, _ = precision_recall_fscore_support(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"], average='micro')
            self.test_precision_weighted, self.test_recall_weighted, self.test_f1_weighted, _ = precision_recall_fscore_support(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"], average='weighted')

        else:

            # Calculate accuracy
            self.validation_accuracy = accuracy_score(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"])

            # Calculate precision, recall, and F1-score
            self.validation_precision_macro, self.validation_recall_macro, self.validation_f1_macro, _ = precision_recall_fscore_support(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"], average='macro')
            self.validation_precision_micro, self.validation_recall_micro, self.validation_f1_micro, _ = precision_recall_fscore_support(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"], average='micro')
            self.validation_precision_weighted, self.validation_recall_weighted, self.validation_f1_weighted, _ = precision_recall_fscore_support(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"], average='weighted')

        print(f"Accuracy: {self.test_accuracy}")
        print()

        print(f"Weighted-Averaged Precision: {self.test_precision_weighted}")
        print(f"Weighted-Averaged Recall: {self.test_recall_weighted}")
        print(f"Weighted-Averaged F1-score: {self.test_f1_weighted}")

    def __getitem__(self, key):
        return getattr(self, key)
