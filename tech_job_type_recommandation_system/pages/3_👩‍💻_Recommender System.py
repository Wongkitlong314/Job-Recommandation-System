import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from io import StringIO
from joblib import dump, load


# Header
text = """
<div style="text-align: center;">
    <h1>Job Recommendation App</h1>
</div>
"""
st.markdown(text, unsafe_allow_html=True)


st.sidebar.markdown("# Job Recommendation 👩‍💻")
st.sidebar.write("Please choose your input type:")
# Images
st.write(" ")
col1, col2 = st.columns(2)

with col1:
    st.image("2.svg")
    st.write(" ")
with col2:
    st.image("3.svg")

# Read input text
def gpt_read_cv(cv):
    client = OpenAI(api_key='YOUR_OPEN_AI_API_KEY')

    sys_prompt = '''
    Be a professional HR manager. You will receive a cv from a candidate.
    You have to judge whether the candidate pose these skills.
    If you are not certain if the candidate pose that skill, write 0, otherwise write 1.
    Format your ouput as the table and fill in the table with the requirement.
    You need to fill all field.
    For coding year of experience, if the candidate does not know any programming language, it should be 0. 
    For coding year of experience, if the candidate does not know any machine learning library or algo, it should be 0. 
    For year calculation, today is 2023 Sept.

    Skill	Value
    Analyse data 1: if have 0: if don't have
    Build Data Infrastructure	1: if have 0: if don't have
    Build Machine Learning Prototypes	1: if have 0: if don't have
    Do Experiment Machine Learning Models	1: if have 0: if don't have 
    Build Machine Learning Service	1: if have 0: if don't have
    Machine Learning Research	1: if have 0: if don't have
    Python	1: if have 0: if don't have
    R	1: if have 0: if don't have
    SQL	1: if have 0: if don't have
    C	1: if have 0: if don't have
    C++	1: if have 0: if don't have
    Java	1: if have 0: if don't have
    JavaScript	1: if have 0: if don't have
    Julia	1: if have 0: if don't have
    Bash	1: if have 0: if don't have
    MATLAB	1: if have 0: if don't have
    Scikit-Learn	1: if have 0: if don't have
    TensorFlow	1: if have 0: if don't have
    Keras	1: if have 0: if don't have
    PyTorch	1: if have 0: if don't have
    Fastai	1: if have 0: if don't have
    XGBoost	1: if have 0: if don't have
    LightGBM	1: if have 0: if don't have
    CatBoost	1: if have 0: if don't have
    Caret	1: if have 0: if don't have
    Tidymodels	1: if have 0: if don't have
    JAX: if have 0: if don't have
    Regression   1: if have 0: if don't have
    decision tree and random forest  1: if have 0: if don't have
    gradient boosting (xgboost, lightgbm)  1: if have 0: if don't have
    Bayesian approach   1: if have 0: if don't have
    Evolutionary approach 1: if have 0: if don't have
    Dense Neural Network 1: if have 0: if don't have
    Convolutional Neural Network 1: if have 0: if don't have
    Generative Adversarial Network 1: if have 0: if don't have
    Recurrent Neural Network 1: if have 0: if don't have
    Transformer Networks (BERT, gpt-3, etc) 1: if have 0: if don't have
    coding year of experience	year, int
    machine learning year of experience 	year, int
    highest obtained education level	categorical: [Bachelor Degree / Master Degree /  Doctor Degree, Other]

    Output requirment
    - format: field: value field2: value2 field3: value3  ...
    - the output feature order should follow the instruction from top to bottom
    - no need any explaination
    '''

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": sys_prompt,
            },                
            {
                  "role": "user",
                "content": cv,
            }
        ],            
        model="gpt-3.5-turbo",
        temperature=0.2
    )
    return chat_completion.choices[0].message.content


# Preprocess the input txt data
def txt_data_pre(text,age,salary,region):
    # Convert string to file-like object 
    data_io = StringIO(response)
    # Create a dictionary to store roles and corresponding values
    role_values = {}
    # Treat each skill in the original data as a key-value pair of a dictionary
    for line in data_io:
        if ":" in line:
            role, value = map(str.strip, line.split(":",1))
            role_values[role] = value

    # Create Pandas DataFrame
    df = pd.DataFrame(list(role_values.items()), columns=['role_title', 'value'])
    df = df.T

    df.columns = ['do_analyze_data','do_build_data_infra','do_build_ml_prototypes',
                  'do_experiment_ml_models','do_build_ml_service','do_ml_research',
              'use_python','use_r', 'use_sql', 'use_c', 'use_c++', 'use_java', 'use_javascript',
              'use_julia', 'use_bash', 'use_matlab', 
              'ml_use_scikit_learn', 'ml_use_tensorflow', 'ml_use_keras',
              'ml_use_pytorch', 'ml_use_fastai', 'ml_use_xgboost', 'ml_use_lightgbm',
                'ml_use_catboost', 'ml_use_caret', 'ml_use_tidymodels', 'ml_use_jax',
              'algo_reg', 'algo_decision_tree',
            'algo_gradient_boosting', 'algo_bayes', 'algo_evo', 'algo_dense_nn',
            'algo_cnn', 'algo_gan', 'algo_rnn', 'algo_trans',
            'year_of_coding_int','year_of_ml_int','highest_edu_lel'
             ]
    df = df.drop('role_title')
    # Feature mapping
    # Deal with education
    df['highest_edu_lv_1. Bachelor’s degree'] = 0
    df['highest_edu_lv_1. Bachelor’s degree'] = np.where(np.any(df['highest_edu_lel'] == 'Bachelor Degree'), 1, 0)

    df['highest_edu_lv_2. Master\'s degree'] = 0
    df['highest_edu_lv_2. Master\'s degree'] = np.where(np.any(df['highest_edu_lel'] == 'Master Degree'), 1, 0)

    df['highest_edu_lv_3. Doctoral degree'] = 0
    df['highest_edu_lv_3. Doctoral degree'] = np.where(np.any(df['highest_edu_lel'] == 'Doctor Degree'), 1, 0)

    df['highest_edu_lv_Other'] = 0
    df['highest_edu_lv_Other'] = np.where(np.any(df['highest_edu_lel'] == 'Doctor Degree'), 1, 0)
    df = df.drop(['highest_edu_lel'], axis=1)

    # Insert do_none
    df['do_none']=1
    df['do_none'] = np.where(np.any(df[['do_analyze_data', 'do_build_data_infra', 'do_build_ml_prototypes', 'do_build_ml_service', 'do_ml_research']] == 0, axis=1), 1, 0)

    # Insert use_none
    df['use_none']=1
    df['use_none'] = np.where(np.any(df[['use_python', 'use_r', 'use_sql', 'use_c', 'use_c++',
                                        'use_java','use_javascript','use_julia','use_bash','use_matlab']] == 0, axis=1), 1, 0)

    # Insert ml_use_none
    df['ml_use_none']=0
    df['ml_use_none'] = np.where(np.any(df[['ml_use_scikit_learn', 'ml_use_tensorflow', 'ml_use_keras', 'ml_use_pytorch', 'ml_use_fastai',
                                        'ml_use_xgboost','ml_use_lightgbm','ml_use_catboost','ml_use_caret','ml_use_tidymodels',
                                            'ml_use_jax']] == 0, axis=1), 1, 0)
    #Insert algo_other & algo_none
    df['algo_none']=1
    df['algo_none'] = np.where(np.any(df[['algo_reg', 'algo_decision_tree', 'algo_gradient_boosting', 'algo_bayes', 'algo_evo',
                                        'algo_dense_nn','algo_cnn','algo_gan','algo_rnn','algo_trans'
                                            ]] == 0, axis=1), 1, 0)
    df['age_int'] = age
    df['salary_int'] = salary
    df['region'] = region

    # Standarized Salary 
    average_salary_by_country = df.groupby('region')['salary_int'].mean().reset_index()
    average_salary_by_country.columns = ['region', 'avgsalary']
    # # Merge the average salary back into the original DataFrame
    df = pd.merge(df, average_salary_by_country, on='region')
    # # Standardize salary based on the average salary in each country
    df['standardized_salary'] = float(df['salary_int']) / df['avgsalary']
   
    # Drop useless features
    df = df.drop(columns = ['salary_int','region','avgsalary'])

    # prepare the columns
    test_columns = ['do_analyze_data', 'do_build_data_infra', 'do_build_ml_prototypes',
        'do_build_ml_service', 'do_experiment_ml_models', 'do_ml_research',
        'do_none', 'use_python', 'use_r', 'use_sql', 'use_c', 'use_c++',
        'use_java', 'use_javascript', 'use_julia', 'use_bash', 'use_matlab',
        'use_none', 'ml_use_scikit_learn', 'ml_use_tensorflow', 'ml_use_keras',
        'ml_use_pytorch', 'ml_use_fastai', 'ml_use_xgboost', 'ml_use_lightgbm',
        'ml_use_catboost', 'ml_use_caret', 'ml_use_tidymodels', 'ml_use_jax',
        'ml_use_none', 'algo_reg', 'algo_decision_tree',
        'algo_gradient_boosting', 'algo_bayes', 'algo_evo', 'algo_dense_nn',
        'algo_cnn', 'algo_gan', 'algo_rnn', 'algo_trans', 'algo_none',
        'age_int', 'year_of_coding_int', 'year_of_ml_int',
        'highest_edu_lv_1. Bachelor’s degree',
        'highest_edu_lv_2. Master\'s degree',
        'highest_edu_lv_3. Doctoral degree', 'highest_edu_lv_Other',
        'standardized_salary']

    # use reindex to make sure the order of columns is the same as the training data
    input_df = df.reindex(columns=test_columns)

    # Change type to int
    columns_to_convert = input_df.columns[:-1]
    input_df[columns_to_convert] = input_df[columns_to_convert].astype(int)
    return input_df

# feature select 
def user_input_features():
    age_int = st.slider('Age', 18, 100, 30, key = 'age')
    regions_list = ['United States of America', 'Other', 'Germany', 'India', 'Russia', 'UK', 'Brazil', 'Nigeria', 'Spain', 'Japan', 'None']
    region_selected = st.selectbox('Expected Working Country', regions_list)
    salary_int = st.slider('Expected Salary', 0, 1000000, 50000)
    average_salary_by_country = {'Brazil': 24991.606170598912, 'Germany': 76176.60550458716, 'India': 20778.81417208966, 'Japan': 48152.56076388889, 'Nigeria': 7719.529085872577, 'Other': 35260.958526796254, 'Russia': 22338.165680473372, 'Spain': 45891.90317195326, 'UK': 81936.95965417867, 'United States of America': 135606.4960629921}
    region_avg_salary = average_salary_by_country.get(region_selected, average_salary_by_country['Other'])
    standardized_salary = salary_int / region_avg_salary
    year_of_coding_int = st.slider('Years of Coding', 0, 50, 5)
    year_of_ml_int = st.slider('Years of ML', 0, 50, 5)
    highest_edu_lv_options = ['1Bachelor’degree', '2Master\'s degree', '3Doctoral degree', 'Other']
    highest_edu_lv_selected = st.selectbox('Highest Education Level', highest_edu_lv_options)
    do_options = ['Analyse data', 'Build Data Infrastructure', 'Build Machine Learning Prototypes', 'Build Machine Learning Service', 'Do Experiment Machine Learning Models', 'Do Machine Learning Research']
    st.write('What do you do?')
    do_selected = [option for option in do_options if st.checkbox(option)]
    use_options = ['python', 'r', 'sql', 'c', 'c++', 'java', 'javascript', 'julia', 'bash', 'matlab']
    st.write('What do you use?')
    use_selected = [option for option in use_options if st.checkbox(option)]
    ml_use_options = ['scikit_learn', 'tensorflow', 'keras', 'pytorch', 'fastai', 'xgboost', 'lightgbm', 'catboost', 'caret', 'tidymodels', 'jax']
    st.write('What ML frameworks do you use?')
    ml_use_selected = [option for option in ml_use_options if st.checkbox(option)]
    algo_options = ['Regression', 'Decision tree and random forest', 'Gradient boosting (xgboost, lightgbm)', 'Bayesian approach', 'Evolutionary approach', 'Dense Neural Network', 'Convolutional Neural Network', 'Generative Adversarial Network', 'Recurrent Neural Network', 'Transformer Networks (BERT, gpt-3, etc)']
    st.write('What algorithms do you use?')
    algo_selected = [option for option in algo_options if st.checkbox(option)]
    
    # Convert selected options into features
    features = {f'do_{option}': int(option in do_selected) for option in do_options}
    features['do_none'] = int(not any(features.values()))
    features.update({f'use_{option}': int(option in use_selected) for option in use_options})
    features['use_none'] = int(not any(features.values()))
    features.update({f'ml_use_{option}': int(option in ml_use_selected) for option in ml_use_options})
    features['ml_use_none'] = int(not any(features.values()))
    features.update({f'algo_{option}': int(option in algo_selected) for option in algo_options})
    features['algo_none'] = int(not any(features.values()))

    features.update({
        'age_int': age_int,
        'standardized_salary': standardized_salary,
        'year_of_coding_int': year_of_coding_int,
        'year_of_ml_int': year_of_ml_int,
    })

    for option in highest_edu_lv_options:
        features[f'highest_edu_lv_{option}'] = int(option == highest_edu_lv_selected)

    return pd.DataFrame(features, index=[0]), region_selected


def create_job_link(region, role_title):
    link = 'https://www.google.com/search?q=' + region.replace(' ', '+')+ '+' + role_title.replace(' ', '+') + '&ibp=htl;jobs&sa=X&ved=2ahUKEwjWyvrFxb2CAxXC4jgGHbV7DlAQudcGKAF6BAgOEC4&sxsrf=AM9HkKlqtAZ4btzm9x0kE5t53n6cjp1Q9A:1699760566091#htivrt=jobs&htidocid=gqDQVJxw4TpaFOo6AAAAAA%3D%3D&fpstate=tldetail' 
    return link

# load model
model = load('classifier_LGBM.joblib')

input_choice = st.sidebar.radio('', ["Manual Input", ":rainbow[Autofill By Upload CV]"]) 
option_select = 1 if input_choice == 'Manual Input' else None
option_txt = 1 if input_choice == ':rainbow[Autofill By Upload CV]' else None
num_recommendation = st.number_input("Number of Recommendation",value=2, step=1,max_value=5,min_value=0)


# Choose to select Feature
if option_select == 1:
    input_df, region = user_input_features()

    # prepare the columns
    test_columns = ['do_analyze_data', 'do_build_data_infra', 'do_build_ml_prototypes',
       'do_build_ml_service', 'do_experiment_ml_models', 'do_ml_research',
       'do_none', 'use_python', 'use_r', 'use_sql', 'use_c', 'use_c++',
       'use_java', 'use_javascript', 'use_julia', 'use_bash', 'use_matlab',
       'use_none', 'ml_use_scikit_learn', 'ml_use_tensorflow', 'ml_use_keras',
       'ml_use_pytorch', 'ml_use_fastai', 'ml_use_xgboost', 'ml_use_lightgbm',
       'ml_use_catboost', 'ml_use_caret', 'ml_use_tidymodels', 'ml_use_jax',
       'ml_use_none', 'algo_reg', 'algo_decision_tree',
       'algo_gradient_boosting', 'algo_bayes', 'algo_evo', 'algo_dense_nn',
       'algo_cnn', 'algo_gan', 'algo_rnn', 'algo_trans', 'algo_none',
       'age_int', 'year_of_coding_int', 'year_of_ml_int',
       'highest_edu_lv_1. Bachelor’s degree',
       'highest_edu_lv_2. Master\'s degree',
       'highest_edu_lv_3. Doctoral degree', 'highest_edu_lv_Other',
       'standardized_salary']

    # use reindex to make sure the order of columns is the same as the training data
    input_df = input_df.reindex(columns=test_columns)
    # st.write(input_df)

    #  predict_proba to get the predication probability
    prediction_proba = model.predict_proba(input_df)

    # Get the indices of the two categories with the highest probability
    top = np.argsort(prediction_proba, axis=1)[:,-num_recommendation:]

    label_mapping = {0 : 'Data Analyst', 1 : 'Data Engineer', 2:'Data Scientist', 3 : 'Engineer (non-software)', 4 : 'Machine Learning Engineer', 5 : 'Manager', 6:'Research Scientist', 7:'Software Engineer',8: 'Statistician',9:'Teacher / professor'}

    # Inverse the label to get the top first
    inverse_label = np.flip(top,axis=1)
    # inverse_label #The label with the highest probability
    inverse_label_int = inverse_label[0].tolist()

    recommendations = [label_mapping[i] for i in inverse_label_int]
    recommendations_output = ', '.join(recommendations)  

    if st.button('Recommend'):
        st.write(f"### Your Top {num_recommendation} Recommendations: {recommendations_output}")
        for predicted_role in recommendations: 
            job_link = create_job_link(region, predicted_role)
            col1, col2 = st.columns([1, 1])
            col1.caption(f"### {predicted_role} Jobs in {region}: ")
            col2.link_button("Search Now", job_link)

# Choose to input text
if option_txt == 1:
    # Text Input    
    age_int = st.slider('Age', 18, 100, 30)
    salary_int = st.slider('Salary', 0, 1000000, 50000)
    region = st.selectbox('Region',('United States of America', 'Other', 'Germany', 'India', 'Russia',
       'UK', 'Brazil', 'Nigeria', 'Spain', 'Japan'))
    st.markdown("### Paste your CV")
    profile = st.text_area(label="")
    button = st.button("Submit", key="button") 

    if button and profile is not None and profile.strip() !="":
        #laoding animation
        with st.spinner('Reading ...'):
            response = gpt_read_cv(profile)
            st.success('Done!')
        
        st.write(response)
        input_df = txt_data_pre(response,age_int,salary_int,region)
        
        
        prediction_proba = model.predict_proba(input_df)

        
        top = np.argsort(prediction_proba, axis=1)[:,-num_recommendation:]

        # create a dictionary to map the label
        label_mapping = {0 : 'Data Analyst', 1 : 'Data Engineer', 2:'Data Scientist', 3 : 'Engineer (non-software)', 4 : 'Machine Learning Engineer', 5 : 'Manager', 6:'Research Scientist', 7:'Software Engineer',8: 'Statistician',9:'Teacher / professor'}

        # Inverse the label to get the top first
        inverse_label = np.flip(top,axis=1)
        # inverse_label
        inverse_label_int = inverse_label[0].tolist()

        recommendations = [label_mapping[i] for i in inverse_label_int]
        recommendations_output = ', '.join(recommendations)  
        
        st.write(f"### Your Top {num_recommendation} Recommendations: {recommendations_output}")
        for predicted_role in recommendations: 
            job_link = create_job_link(region, predicted_role)
            col1, col2 = st.columns([1, 1])
            col1.caption(f"### {predicted_role} Jobs in {region}: ")
            col2.link_button("Search Now", job_link)

    else :
        st.write("Please input your profile")
        

