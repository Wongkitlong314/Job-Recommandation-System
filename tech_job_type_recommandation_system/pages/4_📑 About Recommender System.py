# main_page.py

import streamlit as st

st.set_page_config(
    page_title="Introdctuion",
    page_icon="👋",
)

st.markdown("# About the Recommander System 📑")
st.sidebar.markdown("# About the Recommander System 📑")

intro_text = '''
### Methodology ###

- Data Source: Kaggle Survey 2021, 2022, 2023 
- Classifier: LightGBM  
- Accuracy: Top 2: 0.70; Top 1: 0.48 

### Limiation ###

#### Selection Bias: skewed insights and recommendations ####

- Kaggle user dataset introduces bias as it primarily consists of men and individuals in data science roles.
- Imbalance in representation across job types; fewer instances of managerial, office, and non-data science roles.


#### Survey and Data Collection Limitations ####

- Potential human errors in respondents' survey submissions may introduce inaccuracies.
- Errors by individuals involved in the data collection process can affect the model's overall performance.


#### Model Accuracy & Methodology ####

- The dependent variable in the training dataset only includes the user's current job title, thus failing to capture their complete work history.
- Occupation determination is influenced by multiple factors beyond skillset, not fully captured by the model.
- Some essential skillsets in the workplace are not covered, impacting the comprehensiveness of the model.




'''

st.write(intro_text)

