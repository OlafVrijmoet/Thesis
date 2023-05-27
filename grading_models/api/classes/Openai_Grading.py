
import time
import re

import openai
openai.api_key = "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"

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

  def validation(self):
    """
    Perform validation for the regression grading model.
    """

    # measure performance on validation datasetsplit
    self.measure_performance(
        dataset_split=VALIDATION,
    )

  def make_predictions(self, dataset_split):
    
    # Loop through the sampled dataframe
    for index, row in self.dataset[dataset_split].iterrows():
      
      # Get the predicted points using the grade_student_answer function
      predicted_points = self.grade_student_answer(row=row, model=self.model, index=index)

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
          
          # save predictions
          

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
