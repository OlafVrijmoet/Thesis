
import numpy as np

from sklearn.metrics import mean_squared_error

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

    def model_accuracty(self, df_name):

        # make predictions based on df_name (the df split)
        y_pred = self.model.predict(self.dataset[df_name][self.x_column].values.reshape(-1, 1))

        # add predictions to the df
        self.dataset[df_name]["y_pred"] = y_pred

        # round the y_pred value to the closest whole number between 0 and max_points
        self.dataset[df_name]["pred_points"] = np.round(y_pred).clip(0, self.dataset[df_name]["max_points"])

    def __getitem__(self, key):
        return getattr(self, key)

    