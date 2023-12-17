# page_about.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from data_preprocessing.data_pre import data_preprocessing

# Ignore specific warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("# Feature Trend 📈")
st.sidebar.markdown("# Feature Trend 📈")
st.write(" ")

def plot_trend(data, feature, ylim):
    fig, ax = plt.subplots(figsize=(15, 8))  # Set figure size

    data['all'] = 1
    median_values = data.groupby([feature, 'year', 'gender'])['salary_int'].median().reset_index()
    median_values['median_diff'] = median_values.groupby([feature, 'year'])['salary_int'].transform(lambda x: (x.iloc[0] - x.iloc[1]) / x.iloc[1] * 100)
    if ylim == None:
        ylim = median_values['median_diff'].max()
        

    sns.lineplot(x='year', y='median_diff', hue=feature, data=median_values, ax=ax)
    title_fontsize = 24  # Set title font size
    xlabel_fontsize = 18  # Set x-axis label font size
    ylabel_fontsize = 18  # Set y-axis label font size

    ax.set_title(f'Differences in Median Salaries of Men to Women in {plt_title_dict[feature]}', fontsize=title_fontsize)
    sns.set(style="darkgrid")
    ax.set_xlabel('Year', fontsize=xlabel_fontsize)
    ax.set_ylabel('Difference of Median Salaries(%)', fontsize=ylabel_fontsize)
    ax.set_xticks([2020, 2021, 2022])
    ax.set_ylim(0, ylim)
    ax.legend(loc='upper right')

    plt.show()
    st.pyplot()

def highlight_feature_pt_one(feature):
    if feature == 'all':
        title = "**Observation 1: Gender Pay Gap Trends in the Tech Industry**"
        pay_gap = "**_Global Tech Pay Gap:_**"
        description = "In 2020, the gender pay gap in the tech industry was 250%. By 2022, it decreased to 100%, indicating progress. However, a significant pay gap still remains."

        st.markdown(title)
        st.markdown(pay_gap)
        st.markdown(description)

    elif feature == 'highest_edu_lv':
        title = "**Observation 1: Gender Pay Gap Trends in the Tech Industry**"
        pay_gap = "**_Pay Gap Based on Educational Attainment:_**"
        description = "Initially, the gender pay gap increased from 2020 to 2021. Encouragingly, it notably decreased from 2021 to 2022, suggesting progress towards greater pay equity. However, the pay gap has been consistently widening regarding the highest educational achievements."

        st.markdown(title)
        st.markdown(pay_gap)
        st.markdown(description)

    elif feature == 'region':
        title = "**Observation 1: Gender Pay Gap Trends in the Tech Industry**"
        pay_gap = "**_Pay Gap Based on Geographical Location:_**"
        description = "Russia and Brazil witnessed an expanding gender pay gap in the tech industry. However, in other countries, there was a gradual narrowing of the pay gap. This indicates positive results from efforts to address pay disparities, although challenges persist in specific regions."

        st.markdown(title)
        st.markdown(pay_gap)
        st.markdown(description)


ted_title_dict = {
    'all': 'These 3 years',
    'highest_edu_lv': 'Highest Level of Education',
    'region': 'Region',
}

plt_title_dict = {
    'highest_edu_lv': 'Highest Level of Education',
    'year_of_coding': 'Year of Coding',
    'year_of_ml': 'Year of Doing Machine Learning',
    'role_title': 'Role Title',
    'program_language': 'Programming Language',
    'role_activity': 'Role Activity',
    'machine_learning_lib': 'Machine Learning Library',
    'region': 'Region',
    'all': 'These 3 years'
}


st.sidebar.header("Select Options")
feature_list = ['all', 'highest_edu_lv', 'region']
feature_readable = st.sidebar.selectbox("Select Feature", list(ted_title_dict.values()))

feature = next((key for key, value in ted_title_dict.items() if value == feature_readable), None)
ylim = None
data = data_preprocessing()
plot_trend(data, feature, ylim)

# Add the explanation text below the plot
st.markdown("### Explanation")
highlight_feature_pt_one(feature)

