# main_page.py

import streamlit as st

st.set_page_config(
    page_title="Introdctuion",
    page_icon="👋",
)

st.markdown("# Introduction 🎈")
st.sidebar.markdown("# Introduction 🎈")

intro_text = '''### Job Recommendation Project  ###

#### Exploring the Gender Pay Gap in the Tech Industry: A Data Analytics Investigation ####

In an era of progress and inclusivity, it is imperative to address disparities that persist in various professional domains. Our investigation begins with a meticulous time series analysis, charting the evolution of median salaries across different regions and education levels from 2020 to 2022. 

Furthermore, we scrutinize the intricate interplay between salary and gender, dissecting it by various aspects of qualification. This nuanced examination aims to shed light on the reasons behind the pay gap in this field and explore any potential connections to qualifications.



#### Job Recommendation APP: Bridging Careers in the Tech Landscape  ####


Our Web App goes beyond exploration to offer personalized career recommendations. Users are empowered to input key details such as age, work experience, programming languages, and country, enabling the system to seamlessly recommend suitable job positions. Whether providing insights for career transitions or kickstarting a new venture, our recommendation engine analyzes individual profiles to connect users with tailored job opportunities, complete with information links.

Moreover, for those seeking a hassle-free experience, users can effortlessly upload their resumes. Our system not only recommends positions aligned with their expertise but also supplies relevant job information links based on the associated countries. 





'''

st.write(intro_text)

