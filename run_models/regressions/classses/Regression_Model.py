
import numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

# classes
from performance_tracking.classes.Performance_Row import Performance_Row

# class Regression_Model():

#     def __init__(self, 
#             embedding_model_name, classfication_model_name, dataset_name,
#             dataset, model, x_column, y_column
#         ) -> None:
        
#         # naming
#         self.embedding_model_name = embedding_model_name
#         self.classfication_model_name = classfication_model_name
#         self.dataset_name = dataset_name

#         self.dataset = dataset
#         self.model = model

#         # columns
#         self.x_column = x_column
#         self.y_column = y_column

#         # performance
#         self.performance = Performance_Row(
#             embedding_model_name=embedding_model_name,
#             classfication_model_name=classfication_model_name,
#             dataset_name=dataset_name
#         )

#         self.rmse_test = None
#         self.rmse_validation = None

#         self.test_accuracy = None,
#         self.test_precision_macro = None, 
#         self.test_recall_macro = None, 
#         self.test_f1_macro = None,
#         self.test_precision_micro = None, 
#         self.test_recall_micro = None, 
#         self.test_f1_micro = None,
#         self.test_precision_weighted = None, 
#         self.test_recall_weighted = None, 
#         self.test_f1_weighted = None,
        
#         self.validation_accuracy = None,
#         self.validation_precision_macro = None, 
#         self.validation_recall_macro = None, 
#         self.validation_f1_macro = None,
#         self.validation_precision_micro = None, 
#         self.validation_recall_micro = None, 
#         self.validation_f1_micro = None,
#         self.validation_precision_weighted = None, 
#         self.validation_recall_weighted = None, 
#         self.validation_f1_weighted = None,

#     def train(self):
                
#         self.model = self.model().fit(self.dataset["train_df"][self.x_column].values.reshape(-1, 1), self.dataset["train_df"][self.y_column])
    
#     def test(self):

#         y_pred = self.model.predict(self.dataset["test_df"][self.x_column].values.reshape(-1, 1))

#         # get average max points
#         avg_max_points = self.dataset["test_df"]["max_points"].mean()

#         self.mean_squared_error(test=True, y_ground_truth=self.dataset["test_df"][self.y_column], y_predicted=y_pred, avg_max_points=avg_max_points)

#     def mean_squared_error(self, test: bool, y_ground_truth, y_predicted, avg_max_points):

#         rmse = np.sqrt(mean_squared_error(y_ground_truth, y_predicted)) * avg_max_points

#         if test == True:

#             self.rmse_test = rmse
        
#         else:

#             self.rmse_validation = rmse

#     def model_accuracty(self, test: bool, df_name):

#         # make predictions based on df_name (the df split)
#         y_pred = self.model.predict(self.dataset[df_name][self.x_column].values.reshape(-1, 1))

#         # add predictions to the df as a column
#         self.dataset[df_name]["y_pred"] = y_pred

#         # scale prediction from normalized value back to the orgininal points
#         self.dataset[df_name]["pred_points"] = self.dataset[df_name]["y_pred"] * self.dataset[df_name]["max_points"]

#         # round to closest round number and convert from float to int
#         self.dataset[df_name]["pred_points"] = self.dataset[df_name]["pred_points"].round()
#         self.dataset[df_name]["pred_points"] = self.dataset[df_name]["pred_points"].astype(int)

#         if test == True:

#             # Calculate accuracy
#             self.test_accuracy = accuracy_score(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"])

#             # Calculate precision, recall, and F1-score
#             self.test_precision_macro, self.test_recall_macro, self.test_f1_macro, _ = precision_recall_fscore_support(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"], average='macro')
#             self.test_precision_micro, self.test_recall_micro, self.test_f1_micro, _ = precision_recall_fscore_support(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"], average='micro')
#             self.test_precision_weighted, self.test_recall_weighted, self.test_f1_weighted, _ = precision_recall_fscore_support(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"], average='weighted')

#         else:

#             # Calculate accuracy
#             self.validation_accuracy = accuracy_score(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"])

#             # Calculate precision, recall, and F1-score
#             self.validation_precision_macro, self.validation_recall_macro, self.validation_f1_macro, _ = precision_recall_fscore_support(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"], average='macro')
#             self.validation_precision_micro, self.validation_recall_micro, self.validation_f1_micro, _ = precision_recall_fscore_support(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"], average='micro')
#             self.validation_precision_weighted, self.validation_recall_weighted, self.validation_f1_weighted, _ = precision_recall_fscore_support(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"], average='weighted')

#         print(f"Accuracy: {self.test_accuracy}")
#         print()

#         print(f"Weighted-Averaged Precision: {self.test_precision_weighted}")
#         print(f"Weighted-Averaged Recall: {self.test_recall_weighted}")
#         print(f"Weighted-Averaged F1-score: {self.test_f1_weighted}")

#     def __getitem__(self, key):
#         return getattr(self, key)

#     def save(self):

#         self.performance.save()

class Regression_Model():

    def __init__(self, 
            embedding_model_name, classfication_model_name, dataset_name,
            dataset, model, x_column, y_column
        ) -> None:
        
        # naming
        self.embedding_model_name = embedding_model_name
        self.classfication_model_name = classfication_model_name
        self.dataset_name = dataset_name

        self.dataset = dataset
        self.model = model

        # columns
        self.x_column = x_column
        self.y_column = y_column

        # performance
        self.performance = Performance_Row(
            embedding_model_name=embedding_model_name,
            classfication_model_name=classfication_model_name,
            dataset_name=dataset_name
        )

    def train(self):
                
        self.model = self.model().fit(self.dataset["train_df"][self.x_column].values.reshape(-1, 1), self.dataset["train_df"][self.y_column])
    
    def test(self):

        y_pred = self.model.predict(self.dataset["test_df"][self.x_column].values.reshape(-1, 1))

        # get average max points
        avg_max_points = self.dataset["test_df"]["max_points"].mean()

        self.mean_squared_error(test=True, y_ground_truth=self.dataset["test_df"][self.y_column], y_predicted=y_pred, avg_max_points=avg_max_points)

    def mean_squared_error(self, test: bool, y_ground_truth, y_predicted, avg_max_points):

        rmse = np.sqrt(mean_squared_error(y_ground_truth, y_predicted)) * avg_max_points

        if test:
            self.performance['rmse'] = rmse
        else:
            self.performance['rmse'] = rmse

    def model_accuracy(self, test: bool, df_name):

        # make predictions based on df_name (the df split)
        y_pred = self.model.predict(self.dataset[df_name][self.x_column].values.reshape(-1, 1))

        # add predictions to the df as a column
        self.dataset[df_name]["y_pred"] = y_pred

        # scale prediction from normalized value back to the original points
        self.dataset[df_name]["pred_points"] = self.dataset[df_name]["y_pred"] * self.dataset[df_name]["max_points"]

        # round to closest round number and convert from float to int
        self.dataset[df_name]["pred_points"] = self.dataset[df_name]["pred_points"].round()
        self.dataset[df_name]["pred_points"] = self.dataset[df_name]["pred_points"].astype(int)

        # Calculate accuracy
        accuracy = accuracy_score(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"])

        # Calculate precision, recall, and F1-score
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"], average='macro')
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"], average='micro')
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(self.dataset[df_name]["assigned_points"], self.dataset[df_name]["pred_points"], average='weighted')

        if test:
            self.performance['accuracy'] = accuracy
            self.performance['precision_macro'] = precision_macro
            self.performance['recall_macro'] = recall_macro
            self.performance['f1_macro'] = f1_macro
            self.performance['precision_micro'] = precision_micro
            self.performance['recall_micro'] = recall_micro
            self.performance['f1_micro'] = f1_micro
            self.performance['precision_weighted'] = precision_weighted
            self.performance['recall_weighted'] = recall_weighted
            self.performance['f1_weighted'] = f1_weighted
        else:
            self.performance['accuracy'] = accuracy
            self.performance['precision_macro'] = precision_macro
            self.performance['recall_macro'] = recall_macro
            self.performance['f1_macro'] = f1_macro
            self.performance['precision_micro'] = precision_micro
            self.performance['recall_micro'] = recall_micro
            self.performance['f1_micro'] = f1_micro
            self.performance['precision_weighted'] = precision_weighted
            self.performance['recall_weighted'] = recall_weighted
            self.performance['f1_weighted'] = f1_weighted

        print(f"Accuracy: {accuracy}")
        print()
        print(f"Weighted-Averaged Precision: {precision_weighted}")
        print(f"Weighted-Averaged Recall: {recall_weighted}")
        print(f"Weighted-Averaged F1-score: {f1_weighted}")

    def __getitem__(self, key):
        return getattr(self, key)

    def save(self):

        self.performance.save()
