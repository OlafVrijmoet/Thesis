
class Standardize_Df_Naming:

    def __init__(
            self,
            df,

            row_id,
            question,
            question_id,
            
            student_answer,
            reference_answer,
            assigned_points,
            
            max_points,
            domain,
        ):

            self.df = df

            self.row_id = row_id
            self.question = question
            self.question_id = question_id
            
            self.student_answer = student_answer
            self.reference_answer = reference_answer
            self.assigned_points = assigned_points
            
            self.max_points = max_points
            self.domain = domain
        
