
# libaries
import pandas as pd

# classes
from data.standardized.classes.Standardize_Df_Naming import Standardize_Df_Naming
from data.standardized.classes.Relevant_data import Relevant_data

# services
from data.standardized.services.retrieve_relevant_columns import retrieve_relevant_columns

# print
from services.printing.print_chapter import print_sub_chapter_start, print_sub_chapter_end

# constants
from data.standardized.constants import *

def standardize():

    print_sub_chapter_start(CHAPTER_1)

    # import data
    ASAP_sas = pd.read_csv(ASAP_SAS_RAW)
    beetle = pd.read_csv(BEETLE_RAW)
    neural_course = pd.read_csv(NN_COURSE_RAW)
    sciEntsBank = pd.read_csv(SCI_ENTS_BANK_RAW)
    texas = pd.read_csv(TEXAS_RAW)

    print_sub_chapter_end(CHAPTER_1)

    print_sub_chapter_start(CHAPTER_2)
    # defining dataset setup classes
    ASAP_sas_class = Standardize_Df_Naming(
        df=ASAP_sas,
        df_name="ASAP_sas",
    
        graders=["Score1", "Score2"],
        domain_per_question=[
            "science",
            "science",
            "english_language_arts",
            "english_language_arts",
            "biology",
            "biology",
            "english",
            "english",
            "english",
            "science"
        ],

        row_id=Relevant_data(name="Id"),
        question=Relevant_data(name="question", column=False, value=None),
        question_id=Relevant_data(name="EssaySet"),
        
        student_answer=Relevant_data(name="EssayText"),
        reference_answer=Relevant_data(name="ref_answer", column=False, value=None),
        assigned_points=Relevant_data(name="assigned_points"),
        
        max_points=Relevant_data(name="max_points", column=False, value=None),
        domain=Relevant_data(name="domain", column=False, value=None),
    )

    beetle_class = Standardize_Df_Naming(
        df=beetle,
        df_name="beetle",

        graders=None,
        domain_per_question=None,

        row_id=Relevant_data(name="row_id"),
        question=Relevant_data(name="question"),
        question_id=Relevant_data(name="question_id"),
        
        student_answer=Relevant_data(name="student_answer"),
        reference_answer=Relevant_data(name="reference_answer"),
        assigned_points=Relevant_data(name="assigned_points"),
        
        max_points=Relevant_data(name="max_points"),
        domain=Relevant_data(name="domain"),
    )

    neural_class = Standardize_Df_Naming(
        df=neural_course,
        df_name="neural_course",

        graders=None,
        domain_per_question=None,

        row_id=Relevant_data(name="Unnamed: 0"),
        question=Relevant_data(name="question"),
        question_id=Relevant_data(name="question_id"),
        
        student_answer=Relevant_data(name="student_answer"),
        reference_answer=Relevant_data(name="ref_answer"),
        assigned_points=Relevant_data(name="grades_round"),
        
        max_points=Relevant_data(name="max_points", column=False, value=2),
        domain=Relevant_data(name="domain", column=False, value="neural_networks"),
    )

    sciEntsBank_class = Standardize_Df_Naming(
        df=sciEntsBank,
        df_name="sciEntsBank",

        graders=None,
        domain_per_question=None,

        row_id=Relevant_data(name="row_id"),
        question=Relevant_data(name="question"),
        question_id=Relevant_data(name="question_id"),
        
        student_answer=Relevant_data(name="student_answer"),
        reference_answer=Relevant_data(name="reference_answer"),
        assigned_points=Relevant_data(name="assigned_points"),
        
        max_points=Relevant_data(name="max_points"),
        domain=Relevant_data(name="domain"),
    )

    texas_class = Standardize_Df_Naming(
        df=texas,
        df_name="texas",

        graders=None,
        domain_per_question=None,

        row_id=Relevant_data(name="index", column=False),
        question=Relevant_data(name="question"),
        question_id=Relevant_data(name="id"),
        
        student_answer=Relevant_data(name="student_answer"),
        reference_answer=Relevant_data(name="desired_answer"),
        assigned_points=Relevant_data(name="score_avg"),
        
        max_points=Relevant_data(name="max_points", column=False, value=5),
        domain=Relevant_data(name="domain", column=False, value="science"),
    )
    
    print_sub_chapter_end(CHAPTER_2)
    print_sub_chapter_start(CHAPTER_3)

    # standardize & save datasets
    ASAP_sas_class.standardize_df()
    beetle_class.standardize_df()
    neural_class.standardize_df()
    sciEntsBank_class.standardize_df()
    texas_class.standardize_df()

    print_sub_chapter_end(CHAPTER_3)
