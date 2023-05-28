
import time
import re

import openai
openai.api_key = "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"

import numpy as np

# services
from services.save import save

# classes
from classes.Grading_Model import Grading_Model
from performance_tracking.classes.Performance_Row import Performance_Row

# constants
from performance_tracking.constants import ALL, TRAIN, TEST, VALIDATION

class Openai_Grading(Grading_Model):

  def __init__(self,
    # parent
    model, dataset, measurement_settings,

    # child
    y_column,

    # logs
    logs

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
    
    self.logs = logs

    # saving predictions
    self.y_pred = []

    # make y_pred of None's of length validation set - maybe even in df or np array!?

  def validation(self):
    """
    Perform validation for the regression grading model.
    """

    # measure performance on validation datasetsplit
    self.measure_performance(
        dataset_split=VALIDATION,
    )

  def make_predictions(self, dataset_split):

    # make sure it starts from the index given
    start_index = self.logs.get_index_df(self.dataset["name"])
    
    # Loop through the sampled dataframe from the start_index
    for index, row in self.dataset[dataset_split].iloc[start_index:].iterrows():
        # Get the predicted points using the grade_student_answer function
        predicted_points = self.grade_student_answer(row=row, model=self.model, index=index)

        # save predicted_points in self.y_pred at index row
        self.y_pred.append(predicted_points)

    # return self.y_pred
    return self.y_pred

  def grade_student_answer(row, model, shots, index):

    # Parameters for exponential backoff
    X = 5
    k = 2
    max_attempts = 5

    instruction_line = "Grade the following Student answer based on the Reference answer. The grade should be a howl number."

    messages = [
        {
            "role": "system",
            "content": "You are an AI trained to grade student answers based on a reference answer. Please return a single hwol number."
        },
    ]

    for i in range(1, shots + 1):
        messages.append({
            "role": "system",
            "content": f"""
                {instruction_line}
                Student answer: {row[f'student_answer_{i}']}\n
                Reference answer: {row[f'reference_answer_{i}']}\n
                Grade out of {row['max_points']}: {row[f'assigned_points_{i}']}\n\n
            """
        })

    messages.append({
        "role": "system",
        "content": f"""
            {instruction_line}
            Student answer: {row['student_answer']}\n
            Reference answer: {row['reference_answer']}\n
            Grade out of {row['max_points']}: 
        """
    })

    for attempt in range(max_attempts):
      try:
        response = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=3, n=1, stop=None, temperature=0.5)
        
        # If the API call is successful, we break the loop and don't retry
        break
      except Exception as e:

        print(f"Error on attempt {attempt + 1}: {str(e)}")
        
        # If we've reached max attempts, re-raise the exception
        if attempt + 1 == max_attempts:
          
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
              shots=self.shots,
              epochs=self.epochs,

              dataset_split=dataset_split,

              # duplicates handeling
              settings_performance_tracking=self.measurement_settings.settings_performance_tracking
          )

          run_id = performance_tracking.current_row_id()

          # save run_id df with y_pred values up till now
            # !maybe just use the tracking df! Rather than creating a new one!

            # get length of dataset
            # create a list of Nan's of that length

            # from index 0 fill with self.y_pred and the rest are Nan's

            # save it to the tracking dataset


          # save progress up till now
          self.logs.update_index_df(df_name=self.dataset.name, index=index)

          raise
        
        else:
            # Sleep before next attempt
            time.sleep(X + (attempt ** k))
  
    content = response.choices[0].message['content'].strip()

    # This regex pattern finds float numbers in a string
    float_number_pattern = r"[-+]?[0-9]*\.?[0-9]+"
    numbers = re.findall(float_number_pattern, content)
    
    if numbers:                
        predicted_points = int(round(float(numbers[0])))
    else:
        print("Not valid input!")
        predicted_points = 0

    return predicted_points
