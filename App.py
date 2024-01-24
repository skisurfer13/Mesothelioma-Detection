import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
from PIL import Image
from sklearn import preprocessing



#App description
st.markdown('''
#  Mesothelioma Detector 
This app detects if you have Mesothelioma based on Machine Learning!
- Dataset Creators: Abdullah Cetin Tanrikulu from Dicle University, Faculty of Medicine, Department of Chest Diseases, 21100 Diyarbakir, Turkey
- Orhan Er from Bozok University, Faculty of Engineering, Department of Electrical and Electronics Eng., 66200 Yozgat, Turkey
- Note: User inputs are taken from the sidebar. It is located at the top left of the page (arrow symbol). The values of the parameters can be changed from the sidebar.  
''')
st.write('---')

df = pd.read_csv(r'Mesothelioma data set.csv')

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

dfnew=clean_dataset(df)
#titles
st.sidebar.header('Patient Data')
st.subheader('Training Dataset')
st.write(dfnew.describe())



#train data. Fun!
x = dfnew.drop(['Outcome'], axis = 1)
y = dfnew.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.6, random_state = 0)
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y_train)


#User reports
def user_report():
  Age = st.sidebar.slider('Age', 0,100, 54)
  Platelet_Count = st.sidebar.slider('Platelet Count', 0,3500, 315 )
  Blood_Lactic_Dehydrogenise = st.sidebar.slider('Blood Lactic Dehydrogenise', 0,1000, 20 )
  Alkaline_Phosphatise = st.sidebar.slider('Alkaline Phosphatise', 0,500, 92 )
  Total_Protein = st.sidebar.slider('Total Protein', 0.0,10.0, 5.1 )
  Albumin = st.sidebar.slider('Albumin', 0.0,8.0, 1.1 )
  Glucose = st.sidebar.slider('Glucose', 0,500, 10 )
  Pleural_Lactic_Dehydrogenise = st.sidebar.slider('Pleural Lactic Dehydrogenise', 0,8000, 5 )
  Pleural_Protein = st.sidebar.slider('Pleural Protein', 0.0,8.0, 6.5 )
  Pleural_Albumin = st.sidebar.slider('Pleural Albumin', 0.0,6.0, 4.2 )
  Pleural_Glucose = st.sidebar.slider('Pleural Glucose', 0,120, 44)
  Creactive_Protein = st.sidebar.slider('C-reactive Protein', 0,120, 6)
  
  
  
  
  
  
  user_report_data = {
      'Age':Age,
      'Platelet_Count':Platelet_Count,
      'Blood_Lactic_Dehydrogenise':Blood_Lactic_Dehydrogenise,
      'Alkaline_Phosphatise':Alkaline_Phosphatise,
      'Total_Protein':Total_Protein,
      'Albumin':Albumin,
      'Glucose':Glucose,
      'Pleural_Lactic_Dehydrogenise':Pleural_Lactic_Dehydrogenise,
      'Pleural_Protein':Pleural_Protein,
      'Pleural_Albumin':Pleural_Albumin,
      'Pleural_Glucose':Pleural_Glucose,
      'Creactive_Protein':Creactive_Protein,
        
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data





user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)





rf  = RandomForestClassifier()
rf.fit(x_train, training_scores_encoded)
user_result = rf.predict(user_data)



#Visualizations, this is where the beauty begins.
st.title('Graphical Patient Report')



if user_result[0]==0:
  color = 'blue'
else:
  color = 'red'

#Good old glucose
st.header('Platelet Count Value Graph (Yours vs Others)')
fig_Radius = plt.figure()
ax3 = sns.scatterplot(x = 'Age', y = 'Platelet_Count', data = df, hue = 'Outcome' , palette='Purples')
ax4 = sns.scatterplot(x = user_data['Age'], y = user_data['Platelet_Count'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,3500,175))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Radius)


#Insulin
st.header('Blood Lactic Dehydrogenise Value Graph (Yours vs Others)')
fig_Texture = plt.figure()
ax9 = sns.scatterplot(x = 'Age', y = 'Blood_Lactic_Dehydrogenise', data = df, hue = 'Outcome', palette='rainbow')
ax10 = sns.scatterplot(x = user_data['Age'], y = user_data['Blood_Lactic_Dehydrogenise'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,1000,50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Texture)


#Famous saying BP
st.header('Alkaline Phosphatise Value Graph (Yours vs Others)')
fig_Perimeter = plt.figure()
ax5 = sns.scatterplot(x = 'Age', y = 'Alkaline_Phosphatise', data = df, hue = 'Outcome', palette='Blues')
ax6 = sns.scatterplot(x = user_data['Age'], y = user_data['Alkaline_Phosphatise'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,500,25))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Perimeter)


#Did'nt even know this before nutrition training 
st.header('Total Protein Value Graph (Yours vs Others)')
fig_Area = plt.figure()
ax11 = sns.scatterplot(x = 'Age', y = 'Total_Protein', data = df, hue = 'Outcome', palette='Greens')
ax12 = sns.scatterplot(x = user_data['Age'], y = user_data['Total_Protein'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,10,1))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Area)


#Something new, cool
st.header('Albumin Value Graph (Yours vs Others)')
fig_Smoothness = plt.figure()
ax13 = sns.scatterplot(x = 'Age', y = 'Albumin', data = df, hue = 'Outcome', palette='rocket')
ax14 = sns.scatterplot(x = user_data['Age'], y = user_data['Albumin'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0.0,8.0,0.5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Smoothness)


#Don't even know how thats related to diabetes.The dataset was females only though
st.header('Glucose count Graph (Yours vs Others)')
fig_Compactness = plt.figure()
ax1 = sns.scatterplot(x = 'Age', y = 'Glucose', data = df, hue = 'Outcome', palette = 'magma')
ax2 = sns.scatterplot(x = user_data['Age'], y = user_data['Glucose'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,500,25))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Compactness)


#Wonder how people measure that 
st.header('Pleural Lactic Dehydrogenise Value Graph (Yours vs Others)')
fig_Concavity = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'Pleural_Lactic_Dehydrogenise', data = df, hue = 'Outcome', palette='Reds')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['Pleural_Lactic_Dehydrogenise'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,8000,400))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Concavity)



st.header('Pleural Protein Value Graph (Yours vs Others)')
fig_Concavepoints = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'Pleural_Protein', data = df, hue = 'Outcome', palette='mako')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['Pleural_Protein'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0.0,8.0,0.5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Concavepoints)




st.header('Pleural Albumin Value Graph (Yours vs Others)')
fig_Symmetry = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'Pleural_Albumin', data = df, hue = 'Outcome', palette='flare')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['Pleural_Albumin'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0.0,6.0,0.5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_Symmetry)


st.header('Pleural Glucose Value Graph (Yours vs Others)')
fig_FractalDimension = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'Pleural_Glucose', data = df, hue = 'Outcome', palette='crest')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['Pleural_Glucose'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,120,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_FractalDimension)

st.header('C-reactive Protein Value Graph (Yours vs Others)')
fig_FractalDimension = plt.figure()
ax7 = sns.scatterplot(x = 'Age', y = 'Creactive_Protein', data = df, hue = 'Outcome', palette='crest')
ax8 = sns.scatterplot(x = user_data['Age'], y = user_data['Creactive_Protein'], s = 150, color = color)
plt.xticks(np.arange(0,100,5))
plt.yticks(np.arange(0,120,10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_FractalDimension)








#Finally!
st.subheader('Your Report: ')
output=''
if user_result[0]==0:
  output = 'Congratulations, you do not have  Mesothelioma'
else:
  output = 'Unfortunately, you do have Mesothelioma'
st.title(output)



st.write("This dataset is also available on the UC Irvine Machine Learning Repository")
st.write("Dataset Citation: An approach based on probabilistic neural network for diagnosis of Mesothelioma's disease By: Er, Orhan; Tanrikulu, Abdullah Cetin; Abakay, Abdurrahman; et al. COMPUTERS & ELECTRICAL ENGINEERING Volume: 38 Issue: 1 Pages: 75-81 Published: JAN 2012")




#Most important for users
st.subheader('Lets raise awareness for Mesothelioma and show our support for Mesothelioma awareness and help many patients around the world.')
st.write("Mesothelioma Awareness Day: 26 September")



st.write("Disclaimer: This is just a learning project based on one particular dataset so please do not depend on it to actually know if you have Mesothelioma or not. It might still be a false positive or false negative. A doctor is still the best fit for the determination of such diseases.")
