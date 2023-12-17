# Salary and Gender.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing.data_pre import data_preprocessing
# Ignore specific warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("# Salary Gap 💸")
st.sidebar.markdown("# Salary Gap 💸")


#concat dummy col into one col
convert_dict = {
    'program_language': 'use', # start with use
    'role_activity': 'do',
    'machine_learning_lib': 'ml'
}

# Create a dictionary of reverse mappings
page_title_dict = {
    'highest_edu_lv': 'Highest Level of Education',
    'year_of_coding': 'Year of Coding',
    'year_of_ml': 'Year of Doing Machine Learning',
    'role_title': 'Role Title',
    'program_language': 'Programming Language',
    'role_activity': 'Role Activity',
    'machine_learning_lib': 'Machine Learning Library',
    'region': 'Region',
    # 'all': 'these 3 years'
}

def dummy_to_col(data, feature, convert_dict):
  analysis_data = pd.DataFrame()
  cols = [col for col in data.columns if col.startswith(convert_dict[feature])]
  for col in cols:
    grp_col = ['gender']
    grp_col.append(col)
    plt_data = data.loc[data[col] == 1, ['year', 'gender', 'salary_int', 'region']]
    plt_data[feature] = col
    analysis_data = pd.concat([analysis_data, plt_data])
  return analysis_data

def combine_datasets(period, data):
    start_year, end_year = period
    combined_df = data[(data['year'] >= start_year) & (data['year'] <= end_year)]

    combined_df = combined_df.reset_index(drop=True)
    return combined_df

def plt_salary_gender(data, feature, time_period, region):
    # Generate the combined dataset based on the selected time_period
    plt_data = combine_datasets(time_period, data)

    if region is not None:
        plt_data = plt_data[plt_data['region'] == region]

    # Create the figure and axes for the plot
    fig, ax = plt.subplots(figsize=(15, 8))

    # Define colors for gender categories
    colors = {'Man': 'blue', 'Woman': 'orange'}

    # Create a boxplot with gender as hue
    sns.boxplot(x=feature, y='salary_int', data=plt_data, hue='gender', ax=ax,
                order=sorted(plt_data[feature].apply(str).unique()), showfliers=False,
                palette='deep', color=colors, hue_order=['Man', 'Woman'])

    # Set the plot title based on the selected feature
    title_fontsize = 24
    ax.set_title('Relationship of Salary and %s by Gender (%s)' % (page_title_dict[feature], time_period), fontsize=title_fontsize)

    # Set the labels for y-axis and x-axis
    label_fontsize = 18
    ax.set_ylabel('Salary', fontsize=label_fontsize)
    ax.set_xlabel(page_title_dict[feature], fontsize=label_fontsize)

    # Rotate x-axis labels for better readability
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    # Add legend for gender categories
    ax.legend(loc='upper right')

    # Filter out feature items that do not have both men and women data
    feature_item = plt_data.groupby(feature).gender.nunique()
    feature_item = list(feature_item[feature_item.values == 2].index)
    plt_data = plt_data[plt_data[feature].isin(feature_item)]

    # Calculate the median values for each category and gender
    median_values = plt_data.groupby([feature, 'gender'])['salary_int'].median().reset_index()

    # Calculate the percentage difference of salary between women and men
    gender_median = median_values.groupby(feature)['salary_int'].transform(lambda x: (x.iloc[0] - x.iloc[1]) / x.iloc[1] * 100)

    # Add the percentage difference as a single label on the bars for each category
    for i, category in enumerate(median_values[feature].unique()):
        category_median_diff = gender_median[median_values[feature] == category].iloc[0]

        if category_median_diff < 10:
            color = 'green'
        elif category_median_diff >= 10:
            color = 'red'
        else:
            color = 'black'

        box_props = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4')
        ax.text(i, 0, f"{category_median_diff:.0f}%", ha='center', va='top', bbox=box_props, color=color)

    # Add a text at the bottom of the plot to explain the percentage difference
    plt.figtext(0.1, -0.3, "The Percentage Shown is the Difference of Median Salary of Men to Women.",
                ha="left", fontsize=16, bbox={"facecolor": "white", "alpha": 0.5, "pad": 5})

    # Show the plot
    plt.show()
    st.pyplot()

def visualize_survey_data(data, feature, time_period, region):
    # Generate the combined dataset based on the selected time_period
    combined_df = combine_datasets(time_period, data)

    # set region
    if region is not None:
        combined_df = combined_df[combined_df['region'] == region]

    # Create a categorical variable for the x-axis based on the desired order
    order = sorted(data[feature].apply(str).unique())
    combined_df[feature] = pd.Categorical(combined_df[feature], categories=order, ordered=True)

    # Sort the DataFrame based on the categorical variable
    combined_df = combined_df.sort_values(by=[feature])

    # Define custom colors for the two genders (adjust as needed)
    colors = {'Man': 'blue', 'Woman': 'orange'}

    # Use combined_df to create plots and analyze the data
    plt.figure(figsize=(15, 8))
    ax = sns.histplot(data=combined_df, x=feature, hue='gender', kde=False, common_norm=False,
                      hue_order=['Man', 'Woman'],
                      multiple='dodge', palette='deep', color=colors, alpha=1)

    # Increase the font size of title and labels
    title_fontsize = 24
    label_fontsize = 18
    ax.set_title(f'Distribution of {page_title_dict[feature]} by Gender ({time_period})', fontsize=title_fontsize)
    ax.set_xlabel(page_title_dict[feature], fontsize=label_fontsize)
    ax.set_ylabel('Count', fontsize=label_fontsize)
    ax.tick_params(axis='x', labelrotation=90)  # Rotate x-axis labels for better readability

    plt.show()
    st.pyplot()

def highlight_feature_pt_two(feature):

    title = f"**Observation 2: Factors Influencing the Gender Pay Gap**\n\n***{page_title_dict[feature]}:***"

    if feature == 'highest_edu_lv':
        description = """The gender pay gap varies across educational qualifications, with the highest gaps seen among individuals with doctorate and master's degrees. Surprisingly, those with bachelor's degrees have the lowest pay gap, although all educational categories still face a pay gap of over 90% compared to men."""

    elif feature == 'year_of_coding':
        description = """As coding experience increases, the gender pay gap tends to widen, indicating that experience alone does not mitigate the pay disparity."""

    elif feature == 'year_of_ml':
        description = """The gender pay gap remains relatively stable over years of machine learning experience, suggesting that expertise in machine learning does not strongly influence the pay gap."""

    elif feature == 'role_title':
        description = """Certain role titles such as data scientists, research scientists, software engineers, teachers/professors, and machine learning engineers consistently exhibit higher pay gaps compared to roles like business analysts and data analysts."""

    elif feature == 'program_language':
        description = """The choice of programming language impacts the gender pay gap, with higher gaps observed among individuals using C, JavaScript, MATLAB, Java, and C++. These languages are often associated with software engineering roles, which tend to have higher gender pay gaps."""

    elif feature == 'role_activity':
        description = """Roles classified as "others," typically managerial positions with diverse tasks, display the highest gender pay gap within the tech industry."""

    elif feature == 'machine_learning_lib':
        description = """Professionals using deep learning packages experience a higher gender pay gap, while those utilizing libraries like XGBoost and Scikit-learn encounter smaller gaps. This suggests that among top-tier data scientists and machine learning experts, a significant gender pay gap persists."""

    elif feature == 'region':
        description = """Russia and Brazil witnessed an expanding gender pay gap in the tech industry. However, in other countries, there was a gradual narrowing of the pay gap. This indicates positive results from efforts to address pay disparities, although challenges persist in specific regions."""

    output = f"{title}\n\n{description}"
    st.markdown(output)


# Create a dictionary of reverse mappings
page_title_dict = {
    'highest_edu_lv': 'Highest Level of Education',
    'year_of_coding': 'Year of Coding',
    'year_of_ml': 'Year of Doing Machine Learning',
    'role_title': 'Role Title',
    'program_language': 'Programming Language',
    'role_activity': 'Role Activity',
    'machine_learning_lib': 'Machine Learning Library',
    'region': 'Region',
    # 'all': 'these 3 years'
}

#Time
time_period = st.sidebar.slider("Select Time Period", 2020,2022,(2020,2022))

# Feature
feature_list = ['highest_edu_lv','year_of_coding', 'year_of_ml', 'role_title', 'program_language', 'role_activity', 'machine_learning_lib']
feature_readable = st.sidebar.selectbox("Select Feature", list(page_title_dict.values()))
feature = next((key for key, value in page_title_dict.items() if value == feature_readable), None) 


#Region
regions_list = ['All', 'United States of America', 'Other', 'Germany', 'India', 'Russia', 'UK', 'Brazil', 'Nigeria', 'Spain', 'Japan']
region = st.sidebar.selectbox("Select Region", regions_list)

data = data_preprocessing()
if region == 'All':
    region = None
if feature in ('program_language', 'role_activity', 'machine_learning_lib'):
    plt_data = dummy_to_col(data, feature, convert_dict)
else:
    plt_data = data.copy()


# Call the salary_gender function
st.header("Salary and Gender")
plt_salary_gender(plt_data, feature, time_period, region)  # Call the plt_salary_gender function here

# Add an empty space to separate the two plots
st.write("")

# Call the visualization function
st.header("Distribution of Feature by Gender")
visualize_survey_data(plt_data, feature, time_period, region)

st.markdown("### Explanation")
highlight_feature_pt_two(feature)
