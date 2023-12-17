# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def data_cleaning(): 

    df_2020 = pd.read_csv('dataset\kaggle_survey_2020_responses.csv')
    df_2021 = pd.read_csv('dataset\kaggle_survey_2021_responses.csv')
    df_2022 = pd.read_csv('dataset\kaggle_survey_2022_responses.csv')
    
    # Get the questions' titles
    question_title_2020 =  df_2020[0:1]
    question_title_2021 =  df_2021[0:1]
    question_title_2022 =  df_2022[0:1]

    # add year column
    df_2020['year'] = 2020
    df_2021['year'] = 2021
    df_2022['year'] = 2022

    # # remove the first row, which is the question row
    df_2020 = df_2020.iloc[1:]
    df_2021 = df_2021.iloc[1:]
    df_2022 = df_2022.iloc[1:]

    df_2020.shape[0], df_2021.shape[0],df_2022.shape[0]

    """### construct the combined dataset"""

    data = pd.DataFrame()
    data['completion_time'] = pd.concat([df_2020['Time from Start to Finish (seconds)'], df_2021['Time from Start to Finish (seconds)'], df_2022['Duration (in seconds)']]).astype('int')
    data['year'] = pd.concat([df_2020['year'], df_2021['year'], df_2022['year']])
    data['age'] = pd.concat([df_2020['Q1'], df_2021['Q1'], df_2022['Q2']])
    data['gender'] = pd.concat([df_2020['Q2'], df_2021['Q2'], df_2022['Q3']])
    data['region'] = pd.concat([df_2020['Q3'], df_2021['Q3'], df_2022['Q4']])
    data['highest_edu_lv'] = pd.concat([df_2020['Q4'], df_2021['Q4'], df_2022['Q8']]) #What is the highest level of formal education that you have attained or plan to attain within the next 2 years?

    #occupation
    data['role_title'] = pd.concat([df_2020['Q5'], df_2021['Q5'], df_2022['Q23']]) # Select the title most similar to your current role (or most recent title if retired): - Selected Choice
    data['salary'] = pd.concat([df_2020['Q24'], df_2021['Q25'], df_2022['Q29']]) #What is your current yearly compensation (approximate $USD)?
    #data['company_size'] = pd.concat([df_2020['Q20'], df_2021['Q21'], df_2022['Q25']]) #What is the size of the company where you are employed?
    #data['industry'] = pd.concat([df_2020['Q19'], df_2021['Q20'], df_2022['Q24']]) #2020 no data
    #important activity
    data['do_analyze_data'] = pd.concat([df_2020['Q23_Part_1'], df_2021['Q24_Part_1'], df_2022['Q28_1']]).fillna(0).astype(bool).astype(int)
    data['do_build_data_infra'] = pd.concat([df_2020['Q23_Part_2'], df_2021['Q24_Part_2'], df_2022['Q28_2']]).fillna(0).astype(bool).astype(int)
    data['do_build_ml_prototypes'] = pd.concat([df_2020['Q23_Part_3'], df_2021['Q24_Part_3'], df_2022['Q28_3']]).fillna(0).astype(bool).astype(int)
    data['do_build_ml_service'] = pd.concat([df_2020['Q23_Part_4'], df_2021['Q24_Part_4'], df_2022['Q28_4']]).fillna(0).astype(bool).astype(int)
    data['do_experiment_ml_models'] = pd.concat([df_2020['Q23_Part_5'], df_2021['Q24_Part_5'], df_2022['Q28_5']]).fillna(0).astype(bool).astype(int)
    data['do_ml_research'] = pd.concat([df_2020['Q23_Part_6'], df_2021['Q24_Part_6'], df_2022['Q28_6']]).fillna(0).astype(bool).astype(int)
    data['do_none'] = pd.concat([df_2020['Q23_Part_7'], df_2021['Q24_Part_7'], df_2022['Q28_7']]).fillna(0).astype(bool).astype(int)
    #data['do_other'] = pd.concat([df_2020['Q23_OTHER'], df_2021['Q24_OTHER'], df_2022['Q28_8']]).fillna(0).astype(bool).astype(int)

    # programming skill
    data['year_of_coding'] = pd.concat([df_2020['Q6'], df_2021['Q6'], df_2022['Q11']])
    # programming language
    data['use_python'] = pd.concat([df_2020['Q7_Part_1'], df_2021['Q7_Part_1'], df_2022['Q12_1']]).fillna(0).astype(bool).astype(int)
    data['use_r'] = pd.concat([df_2020['Q7_Part_2'], df_2021['Q7_Part_2'], df_2022['Q12_2']]).fillna(0).astype(bool).astype(int)
    data['use_sql'] = pd.concat([df_2020['Q7_Part_3'], df_2021['Q7_Part_3'], df_2022['Q12_3']]).fillna(0).astype(bool).astype(int)
    data['use_c'] = pd.concat([df_2020['Q7_Part_4'], df_2021['Q7_Part_4'], df_2022['Q12_4']]).fillna(0).astype(bool).astype(int)
    data['use_python'] = pd.concat([df_2020['Q7_Part_1'], df_2021['Q7_Part_1'], df_2022['Q12_1']]).fillna(0).astype(bool).astype(int)
    data['use_c++'] = pd.concat([df_2020['Q7_Part_5'], df_2021['Q7_Part_5'], df_2022['Q12_6']]).fillna(0).astype(bool).astype(int)
    data['use_java'] = pd.concat([df_2020['Q7_Part_6'], df_2021['Q7_Part_6'], df_2022['Q12_7']]).fillna(0).astype(bool).astype(int)
    data['use_javascript'] = pd.concat([df_2020['Q7_Part_7'], df_2021['Q7_Part_7'], df_2022['Q12_8']]).fillna(0).astype(bool).astype(int)
    data['use_julia'] = pd.concat([df_2020['Q7_Part_8'], df_2021['Q7_Part_8'], df_2022['Q12_9']]).fillna(0).astype(bool).astype(int)
    data['use_bash'] = pd.concat([df_2020['Q7_Part_10'], df_2021['Q7_Part_10'], df_2022['Q12_9']]).fillna(0).astype(bool).astype(int)
    data['use_matlab'] = pd.concat([df_2020['Q7_Part_11'], df_2021['Q7_Part_11'], df_2022['Q12_11']]).fillna(0).astype(bool).astype(int)
    data['use_none'] = pd.concat([df_2020['Q7_Part_12'], df_2021['Q7_Part_12'], df_2022['Q12_14']]).fillna(0).astype(bool).astype(int)
    #data['use_other'] = pd.concat([df_2020['Q7_OTHER'], df_2021['Q7_OTHER'], df_2022['Q12_15']]).fillna(0).astype(bool).astype(int)

    # ML skill
    data['year_of_ml'] = pd.concat([df_2020['Q15'], df_2021['Q15'], df_2022['Q16']]) #For how many years have you used machine learning methods?
    #ML framework
    data['ml_use_scikit_learn'] = pd.concat([df_2020['Q16_Part_1'], df_2021['Q16_Part_1'], df_2022['Q17_1']]).fillna(0).astype(bool).astype(int)
    data['ml_use_tensorflow'] = pd.concat([df_2020['Q16_Part_2'], df_2021['Q16_Part_2'], df_2022['Q17_2']]).fillna(0).astype(bool).astype(int)
    data['ml_use_keras'] = pd.concat([df_2020['Q16_Part_3'], df_2021['Q16_Part_3'], df_2022['Q17_3']]).fillna(0).astype(bool).astype(int)
    data['ml_use_pytorch'] = pd.concat([df_2020['Q16_Part_4'], df_2021['Q16_Part_4'], df_2022['Q17_4']]).fillna(0).astype(bool).astype(int)
    data['ml_use_fastai'] = pd.concat([df_2020['Q16_Part_5'], df_2021['Q16_Part_5'], df_2022['Q17_5']]).fillna(0).astype(bool).astype(int)
    data['ml_use_xgboost'] = pd.concat([df_2020['Q16_Part_7'], df_2021['Q16_Part_7'], df_2022['Q17_6']]).fillna(0).astype(bool).astype(int)
    data['ml_use_lightgbm'] = pd.concat([df_2020['Q16_Part_8'], df_2021['Q16_Part_8'], df_2022['Q17_7']]).fillna(0).astype(bool).astype(int)
    data['ml_use_catboost'] = pd.concat([df_2020['Q16_Part_9'], df_2021['Q16_Part_9'], df_2022['Q17_8']]).fillna(0).astype(bool).astype(int)
    data['ml_use_caret'] = pd.concat([df_2020['Q16_Part_12'], df_2021['Q16_Part_12'], df_2022['Q17_9']]).fillna(0).astype(bool).astype(int)
    data['ml_use_tidymodels'] = pd.concat([df_2020['Q16_Part_13'], df_2021['Q16_Part_13'], df_2022['Q17_10']]).fillna(0).astype(bool).astype(int)
    data['ml_use_jax'] = pd.concat([df_2020['Q16_Part_14'], df_2021['Q16_Part_14'], df_2022['Q17_11']]).fillna(0).astype(bool).astype(int)
    #data['ml_use_pytorch_lightning'] = pd.concat([df_2020['Q16_Part_15'], df_2021['Q16_Part_15'], df_2022['Q17_12']]).fillna(0).astype(bool).astype(int)
    # data['ml_use_huggingface'] = pd.concat([df_2020[''], df_2021['Q16_Part_16'], df_2022['Q17_13']]).fillna(0).astype(bool).astype(int)
    data['ml_use_none'] = pd.concat([df_2020['Q16_Part_15'], df_2021['Q16_Part_17'], df_2022['Q17_14']]).fillna(0).astype(bool).astype(int)
    #data['ml_use_other'] = pd.concat([df_2020['Q16_OTHER'], df_2021['Q16_OTHER'], df_2022['Q17_15']]).fillna(0).astype(bool).astype(int)

        #ML algo

    algorithm_order = [
        'algo_reg', 'algo_decision_tree', 'algo_gradient_boosting', 'algo_bayes',
        'algo_evo', 'algo_dense_nn', 'algo_cnn', 'algo_gan', 'algo_rnn',
        'algo_trans', 'algo_none', 'algo_other'
    ]

    data[algorithm_order[0]] = pd.concat([df_2020['Q17_Part_1'], df_2021['Q16_Part_1'], df_2022['Q18_1']]).fillna(0).astype(bool).astype(int)

    # Loop through the remaining algorithms and concatenate the respective columns
    data[algorithm_order[0]] = pd.concat([df_2020['Q17_Part_1'], df_2021['Q17_Part_1'], df_2022['Q18_1']]).fillna(0).astype(bool).astype(int)
    data[algorithm_order[1]] = pd.concat([df_2020['Q17_Part_2'], df_2021['Q17_Part_2'], df_2022['Q18_2']]).fillna(0).astype(bool).astype(int)
    data[algorithm_order[2]] = pd.concat([df_2020['Q17_Part_3'], df_2021['Q17_Part_3'], df_2022['Q18_3']]).fillna(0).astype(bool).astype(int)
    data[algorithm_order[3]] = pd.concat([df_2020['Q17_Part_4'], df_2021['Q17_Part_4'], df_2022['Q18_4']]).fillna(0).astype(bool).astype(int)
    data[algorithm_order[4]] = pd.concat([df_2020['Q17_Part_5'], df_2021['Q17_Part_5'], df_2022['Q18_5']]).fillna(0).astype(bool).astype(int)
    data[algorithm_order[5]] = pd.concat([df_2020['Q17_Part_6'], df_2021['Q17_Part_6'], df_2022['Q18_6']]).fillna(0).astype(bool).astype(int)
    data[algorithm_order[6]] = pd.concat([df_2020['Q17_Part_7'], df_2021['Q17_Part_7'], df_2022['Q18_7']]).fillna(0).astype(bool).astype(int)
    data[algorithm_order[7]] = pd.concat([df_2020['Q17_Part_8'], df_2021['Q17_Part_8'], df_2022['Q18_8']]).fillna(0).astype(bool).astype(int)
    data[algorithm_order[8]] = pd.concat([df_2020['Q17_Part_9'], df_2021['Q17_Part_9'], df_2022['Q18_9']]).fillna(0).astype(bool).astype(int)
    data[algorithm_order[9]] = pd.concat([df_2020['Q17_Part_10'], df_2021['Q17_Part_10'], df_2022['Q18_10']]).fillna(0).astype(bool).astype(int)
    data[algorithm_order[10]] = pd.concat([df_2020['Q17_Part_11'], df_2021['Q17_Part_11'], df_2022['Q18_13']]).fillna(0).astype(bool).astype(int)
    data[algorithm_order[11]] = pd.concat([df_2020['Q17_OTHER'], df_2021['Q17_OTHER'], df_2022['Q18_14']]).fillna(0).astype(bool).astype(int)

    #ml nlp
    #data['algo_no_nlp'] = pd.concat([df_2020['Q19_Part_5'], df_2021['Q19_Part_5'], df_2022['Q20_5']]).fillna(0).astype(bool).astype(int)
    #data['algo_no_cv'] = pd.concat([df_2020['Q18_Part_6'], df_2021['Q18_Part_6'], df_2022['Q19_7']]).fillna(0).astype(bool).astype(int)

    data

    #data.describe().transpose()

    """## Data Cleaning (must run)"""

    #data.info()

    """#### drop data"""

    # filter out non-male or non-female
    data = data[data['gender'].isin(['Man', 'Woman'])]
    # filter out student and unemployed
    data = data[~data['role_title'].isin(['Student', 'Currently not employed'])]
    # filter out people without typing salary
    data = data[~data['salary'].isna()]
    # filter out duration less than 1 min.
    data = data[data['completion_time']>=60]

    """#### label mapping"""

    map_year_of_coding = {
        "I have never written code": "1. 0 years",
        "< 1 years": "2. < 1 years",
        "1-2 years": "3. 1-3 years",
        "1-3 years": "3. 1-3 years",
        "3-5 years": "4. 3-5 years",
        "5-10 years": "5. 5-10 years",
        "10-20 years": "6. 10-20 years",
        "20+ years": "7. 20+ years",
        np.nan: "1. 0 years"
    }

    map_year_of_coding_int = {
        "I have never written code": 0,
        "< 1 years": 1,
        "1-2 years": 3,
        "1-3 years": 3,
        "3-5 years": 5,
        "5-10 years": 10,
        "10-20 years": 20,
        "20+ years": 30,
        np.nan: 0
    }

    map_year_of_ml = {
        "I do not use machine learning methods": "1. 0 years",
        "Under 1 year": "2. < 1 years",
        "1-2 years": "3. 1-3 years",
        "1-3 years": "3. 1-3 years",
        "2-3 years": "4. 2-3 years",
        "3-4 years": "5. 3-4 years",
        "4-5 years": "6. 4-5 years",
        "5-10 years": "7. 5-10 years",
        "10-20 years": "8. 10-20 years",
        "20+ years": "9. 20+ years",
        "20 or more years": "9. 20+ years",
        np.nan: "1. 0 years"
    }

    map_year_of_ml_int = {
        'I do not use machine learning methods': 0,
        '3-4 years': 4,
        '2-3 years': 3,
        'Under 1 year': 1,
        '4-5 years': 5,
        '1-2 years': 2,
        '5-10 years': 10,
        '20 or more years': 30,
        '10-20 years': 20,
        np.nan: 0
    }

    map_age = {
        '30-34': 34,
        '35-39': 39,
        '22-24': 24,
        '55-59': 59,
        '50-54': 54,
        '25-29': 29,
        '18-21': 21,
        '40-44': 44,
        '60-69': 69,
        '45-49': 49,
        '70+': 70
    }

    map_gender = {
        "Man": 1,
        "Woman": 0,
        np.nan: np.nan
    }

    map_region = {
        "United Kingdom of Great Britain and Northern Ireland": "UK",
        np.nan: np.nan
    }

    job_title_mapping = {
        'Data Engineer': 'Data Engineer',
        'Software Engineer': 'Software Engineer',
        'Data Scientist': 'Data Scientist',
        'Data Analyst': 'Data Analyst',
        'Research Scientist': 'Research Scientist',
        'Other': 'Other',
        'Statistician': 'Statistician',
        'Product/Project Manager': 'Manager',
        'Machine Learning Engineer': 'Machine Learning Engineer',
        'Business Analyst': 'Data Analyst', #2022 does not have BA
        'DBA/Database Engineer': 'Data Engineer', #combine with DE
        'Program/Project Manager': 'Manager', #combine with manager
        'Product Manager': 'Manager', #combine with manager
        'Developer Relations/Advocacy': 'Developer Advocate', #combine with Developer Advocate
        'Developer Advocate': 'Developer Advocate',
        'Data Analyst (Business, Marketing, Financial, Quantitative, etc)': 'Data Analyst',
        'Machine Learning/ MLops Engineer': 'Machine Learning Engineer',
        'Engineer (non-software)': 'Engineer (non-software)',
        'Teacher / professor': 'Other', #only 2022
        'Manager (Program, Project, Operations, Executive-level, etc)': 'Manager',
        'Data Administrator': 'Data Administrator',
        'Data Architect': 'Data Architect',
        np.nan: np.nan
    }

    map_company_size_int = {
        "0-49 employees": 49,
        "50-249 employees": 249,
        "250-999 employees": 999,
        "1000-9,999 employees": 9999,
        "10,000 or more employees": 15000,
        np.nan: np.nan
    }

    education_mapping = {
        'Master’s degree': "2. Master's degree",
        'Bachelor’s degree': '1. Bachelor’s degree',
        'Doctoral degree': "3. Doctoral degree",
        'Some college/university study without earning a bachelor’s degree': '0. Unfinished bachelor’s degree',
        'Professional degree': 'Other',
        'I prefer not to answer': 'Other',
        'No formal education past high school': 'Other',
        'Professional doctorate': 'Other',
        np.nan: np.nan
    }

    salary_mapping = {
        '0-999': '<5000',
        '1,000-1,999': '<5000',
        '2,000-2,999': '<5000',
        '3,000-3,999': '<5000',
        '4,000-4,999': '<5000',
        '5,000-7,499': '5,000-9,999',
        '7,500-9,999': '5,000-9,999',
        '10,000-14,999': '10,000-29,999',
        '15,000-19,999': '10,000-29,999',
        '20,000-24,999': '10,000-29,999',
        '25,000-29,999': '10,000-29,999',
        '30,000-39,999': '30,000-49,999',
        '40,000-49,999': '30,000-49,999',
        '50,000-59,999': '50,000-69,999',
        '60,000-69,999': '50,000-69,999',
        '70,000-79,999': '70,000-99,999',
        '80,000-89,999': '70,000-99,999',
        '90,000-99,999': '70,000-99,999',
        '100,000-124,999': '100,000-149,999',
        '125,000-149,999': '100,000-149,999',
        '150,000-199,999': '150,000-249,999',
        '200,000-249,999': '150,000-249,999',
        '250,000-299,999': '250,000-499,999',
        '300,000-499,999': '250,000-499,999',
        '300,000-500,000': '250,000-499,999',
        '500,000-999,999': '>500,000',
        '> 500,000': '>500,000',
        '>1,000,000': '>500,000',
        np.nan: np.nan
    }

    salary_mapping_int = {
        '0-999': 500,
        '1,000-1,999': 1500,
        '2,000-2,999': 2500,
        '3,000-3,999': 3500,
        '4,000-4,999': 4500,
        '5,000-7,499': 6250,
        '7,500-9,999': 8750,
        '10,000-14,999': 12500,
        '15,000-19,999': 17500,
        '20,000-24,999': 22500,
        '25,000-29,999': 27500,
        '30,000-39,999': 35000,
        '40,000-49,999': 45000,
        '50,000-59,999': 55000,
        '60,000-69,999': 65000,
        '70,000-79,999': 75000,
        '80,000-89,999': 85000,
        '90,000-99,999': 95000,
        '100,000-124,999': 112500,
        '125,000-149,999': 137500,
        '150,000-199,999': 175000,
        '200,000-249,999': 225000,
        '250,000-299,999': 275000,
        '300,000-499,999': 400000,
        '300,000-500,000': 400000,
        '500,000-999,999': 750000,
        '> 500,000': 750000,
        '>1,000,000': 750000,
        np.nan: np.nan
    }

    """### column transform"""

    data['age_int'] = data['age'].apply(lambda x : map_age[x])
    data['gender_int'] = data['gender'].apply(lambda x : map_gender[x])
    data['highest_edu_lv'] = data['highest_edu_lv'].apply(lambda x : education_mapping[x])

    #salary
    data['salary'] = data['salary'].str.replace('$', '')
    data['salary_int'] = data['salary'].apply(lambda x : salary_mapping_int[x])
    #data['salary_int_new'] = data['salary'].apply(lambda x : salary_mapping_int_new[x])
    data['salary'] = data['salary'].apply(lambda x : salary_mapping[x])

    #year_of_coding
    data['year_of_coding_int'] = data['year_of_coding'].apply(lambda x : map_year_of_coding_int[x])
    data['year_of_coding'] = data['year_of_coding'].apply(lambda x : map_year_of_coding[x])

    #year_of_ml
    data['year_of_ml_int'] = data['year_of_ml'].apply(lambda x : map_year_of_ml_int[x])
    data['year_of_ml'] = data['year_of_ml'].apply(lambda x : map_year_of_ml[x])

    #region
    data['region'] = data['region'].apply(lambda x : map_region[x] if x in map_region else x)
    most_popular_country = data.groupby('region').count().iloc[:,1].sort_values(ascending = False).head(10).keys()
    data.loc[~data['region'].isin(most_popular_country), 'region'] = 'Other'
    # to dummy
    # dummy_df = pd.get_dummies(data['region'], prefix='region')
    # data = pd.concat([data, dummy_df], axis=1)

    # role title
    data['role_title'] = data['role_title'].apply(lambda x : job_title_mapping[x])
    most_popular_job_title = data.groupby('role_title').count().iloc[:,1].sort_values(ascending = False).head(8).keys()
    data.loc[~data['role_title'].isin(most_popular_job_title), 'role_title'] = 'Other'
    # to dummy
    # dummy_df = pd.get_dummies(data['role_title'], prefix='role_title')
    # data = pd.concat([data, dummy_df], axis=1)

    # highest education level
    most_popular_edu_lv = data.groupby('highest_edu_lv').count().iloc[:,1].sort_values(ascending = False).head(4).keys()
    data.loc[~data['highest_edu_lv'].isin(most_popular_edu_lv), 'highest_edu_lv'] = 'Other'
    # to dummy
    dummy_df = pd.get_dummies(data['highest_edu_lv'], prefix='highest_edu_lv')
    data = pd.concat([data, dummy_df], axis=1)

    #company size
    #data['company_size'] = data['company_size'].apply(lambda x : map_company_size_int[x] if x in map_company_size_int else x)

    # salary standardize among country
    average_salary_by_country = data.groupby('region')['salary_int'].mean().reset_index()
    average_salary_by_country.columns = ['region', 'avgsalary']
    std_salary_by_country = data.groupby('region')['salary_int'].std().reset_index()
    std_salary_by_country.columns = ['region', 'stdsalary']

    # Merge the average salary back into the original DataFrame
    data = pd.merge(data, average_salary_by_country, on='region')
    data = pd.merge(data, std_salary_by_country, on='region')

    # Standardize salary based on the average salary in each country
    data['standardized_salary'] = (data['salary_int'] - data['avgsalary']) / data['std_salary_by_country']

    # drop columns
    drop_col = ['gender_int', 'salary_int', 'std_salary_by_country', 'salary', 'avgsalary', 'completion_time', 'year', 'age', 'gender',  'highest_edu_lv', 'year_of_coding', 'year_of_ml']
    data = data.drop(columns = drop_col)

    return data 




