import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances


header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

@st.cache
def get_data(filename):
    HR_DATA = pd.read_csv(filename)
    return HR_DATA

@st.cache
def get_data2(filename):
    HR_DATA = pd.read_csv(filename)
    return HR_DATA

def get_data3(filename):
    HR_DATA = pd.read_csv(filename)
    return HR_DATA


df_ED = get_data('data/Employee_Designation.csv')
df_ESD = get_data2('data/Employee_Skills_Datset.csv')
df_FED = get_data3('data/Final_Employees_Data.csv') 
del df_FED['Ename']
df_merged = pd.merge(df_ED,df_ESD,how='left',left_on=['Eid'],right_on=['Eid'])
skills_dataset = pd.merge(df_merged,df_FED,how='left',left_on=['Eid'],right_on=['Eid'])

with header:
    st.title('Content-based Recommender System For Skills Dataset')
    st.text("")
    st.text("")
    st.text("")
    st.text('In this project I build a content-based recommender system using Employee_Designation.csv, Employee_Skills_Datset.csv, Final_Employees_Data.csv')

with dataset:
    st.header('Amalgamation of the Employee_Designation.csv, Employee_Skills_Datset.csv, Final_Employees_Data.csv dataset')
    st.text("")
    st.text("")
    st.text("")
    st.text('I got this dataset from Kaggle: https://www.kaggle.com/granjithkumar/it-employees-data-for-project-allocation?select=Employee_Designation.csv')

    st.write(skills_dataset.head())

    st.subheader('Designation distribution')
    state_disribution = pd.DataFrame(skills_dataset['Designation'].value_counts())
    st.bar_chart(state_disribution)

    st.subheader('Area_of_Interest_1 distribution')
    state_disribution = pd.DataFrame(skills_dataset['Area_of_Interest_1'].value_counts())
    st.bar_chart(state_disribution)

    st.subheader('Area_of_Interest_2 distribution')
    state_disribution = pd.DataFrame(skills_dataset['Area_of_Interest_2'].value_counts())
    st.bar_chart(state_disribution)

    st.subheader('Area_of_Interest_3 distribution')
    state_disribution = pd.DataFrame(skills_dataset['Area_of_Interest_3'].value_counts())
    st.bar_chart(state_disribution)

    st.subheader('Language1 distribution')
    state_disribution = pd.DataFrame(skills_dataset['Language1'].value_counts())
    st.bar_chart(state_disribution)

    st.subheader('Language2 distribution')
    state_disribution = pd.DataFrame(skills_dataset['Language2'].value_counts())
    st.bar_chart(state_disribution)

    st.subheader('Language3 distribution')
    state_disribution = pd.DataFrame(skills_dataset['Language3'].value_counts())
    st.bar_chart(state_disribution)

with features:
    st.header('Features')

    st.markdown('* **Input:** I created this feature in order combine all the relevant parameters required for input')

with model_training:
    st.header('Training')
    st.text('Here you can get recommendation for any individual')

    sel_col, disp_col = st.beta_columns(2)

    input_name = sel_col.selectbox('Select any name from dataset', (skills_dataset['Ename']))
    sel_col.subheader('Name Details')
    sel_col.write(skills_dataset.loc[skills_dataset['Ename'] == input_name])

    df = skills_dataset.copy()
    df = df[df['Area_of_Interest_1'].notna()]
    df = df[df['Area_of_Interest_2'].notna()]
    df = df[df['Area_of_Interest_3'].notna()]
    df = df[df['Designation'].notna()]
    df = df[df['Language1'].notna()]
    df = df[df['Language2'].notna()]
    df = df[df['Language3'].notna()]
    df = df[df['Ename'].notna()]
    df['Area_of_Interest_1'] = df['Area_of_Interest_1'].astype(str)
    df['Area_of_Interest_2'] = df['Area_of_Interest_2'].astype(str)
    df['Area_of_Interest_3'] = df['Area_of_Interest_3'].astype(str)
    df['Designation'] = df['Designation'].astype(str)
    df['Language1'] = df['Language1'].astype(str)
    df['Language2'] = df['Language2'].astype(str)
    df['Language3'] = df['Language3'].astype(str)
    df['Ename'] = df['Ename'].astype(str)
    df['Input'] = df[['Designation', 'Area_of_Interest_1', 'Area_of_Interest_2', 
    'Area_of_Interest_3', 'Language1', 'Language2', 'Language3']].apply(lambda x: ' '.join(x), axis = 1)
    metadata = df.copy()
    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')
    #Replace NaN with an empty string
    metadata['Input'] = metadata['Input'].fillna('')
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(metadata['Input'])
    cosine_similarity(tfidf_matrix)
    cosine_distances(tfidf_matrix)
    cosine_model=cosine_similarity(tfidf_matrix)
    cosine_model_df=pd.DataFrame(cosine_model,index=df.Ename,columns=df.Ename)
    cosine_model_df.head()
    def make_recommendations(movie_user_likes):
        return cosine_model_df[movie_user_likes].sort_values(ascending=False)[:10]


    recommender = make_recommendations(input_name)

    disp_col.subheader('Recommendations')
    recommender = recommender.to_frame()
    recommender.reset_index(level=0, inplace=True)
    recommender = recommender.rename(columns={input_name: "Cosine Similarity"})
    disp_col.write(recommender)

    disp_col.subheader('Recommendations Details')
    for i in recommender.Ename:
        disp_col.write(df.loc[df['Ename'] == i])

    


