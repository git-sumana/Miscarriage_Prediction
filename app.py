import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data.csv')

# Split the data into features (X) and target (y)
X = df.drop('Miscarriage', axis=1)
y = df['Miscarriage']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
clf.fit(X_train, y_train)

# Streamlit app
st.title('Miscarriage Prediction')

# Input parameters
age = st.number_input('Age', min_value=0, max_value=100, value=25, help='Enter your age in years')

    # else:
    #     bmi = st.number_input('BMI', min_value=0.0, max_value=50.0, value=25.0, help='Enter your Body Mass Index')
bmi = st.number_input('BMI', min_value=0.0, max_value=50.0, value=25.0, help='Enter your Body Mass Index')
if st.checkbox("I don't know my BMI"):
    weight = st.number_input('Weight (kg)', min_value=0.0, max_value=200.0, value=70.0, help='Enter your weight in kilograms')
    height = st.number_input('Height (cm)', min_value=0.0, max_value=250.0, value=170.0, help='Enter your height in centimeters')
    if st.button('Calculate BMI'):
        bmi = weight / ((height / 100) ** 2)
        st.write('Your BMI is:', bmi)
nmisc = st.number_input('Nmisc', min_value=0, max_value=10, value=1, help='Enter the number of previous miscarriages')
activity = st.number_input('Activity', min_value=0, max_value=10, value=1, help='Enter your activity level (1,2,3) => low, moderate, high')
location = st.number_input('Location', min_value=0, max_value=10, value=3, help='Enter your location (1,2,3) => urban, suburban, rural')
temp = st.number_input('Temp', min_value=0.0, max_value=50.0, value=36.0, help='Enter your body temperature in Celsius')
bpm = st.number_input('BPM', min_value=0, max_value=200, value=100, help='Enter your heart rate in beats per minute')
if st.checkbox("I don't know my Heart Rate (BPM)"):
    st.markdown(
        "You can calculate your Heart Rate using this [Heart Rate Calculator](https://goodcalculators.com/heart-rate-calculator/)."
    )
else:
    bpm = st.number_input('Heart Rate (BPM)', min_value=0, max_value=200, value=70, help='Enter your heart rate in beats per minute')

stress = st.number_input('Stress', min_value=0, max_value=10, value=2, help='Enter your stress level (1,2,3) => low, moderate, high')
bp = st.number_input('BP', min_value=0, max_value=10, value=1, help='Enter your blood pressure level (1,2,3) => low, normal, high')

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Button to make prediction
if st.button('Predict Miscarriage'):
    input_data = pd.DataFrame([[age, bmi, nmisc, activity, location, temp, bpm, stress, bp]], columns=X.columns)
    prediction = clf.predict(input_data)
    if prediction[0] == 1:
        st.write('The model predicts a miscarriage.')
        st.write('Here are some suggestions for improving your health:')
        st.markdown('* Maintain a healthy diet rich in fruits, vegetables, lean proteins, and whole grains.')
        st.markdown('* Regular physical activity can help maintain a healthy weight and overall health.')
        st.markdown('* Avoid smoking and limit alcohol consumption.')
        st.markdown('* Regular check-ups with your healthcare provider can help detect any potential health issues early.')
        # Feature importance
        st.subheader('Feature Importance')
        importance = pd.DataFrame({'feature': X_train.columns, 'importance': clf.feature_importances_})
        importance = importance.sort_values('importance', ascending=False)
        st.bar_chart(importance.set_index('feature'))

            # Line plot of user's input
        st.subheader('Your Input')
        user_input = pd.DataFrame([[age, bmi, nmisc, activity, location, temp, bpm, stress, bp]], columns=X.columns)
        st.line_chart(user_input.T)
        st.markdown('* Mental health is just as important as physical health. Consider activities like meditation or yoga to reduce stress.')
        st.write('Accuracy: ', accuracy)
    else:
        st.write('The model does not predict a miscarriage.')
        st.write('Accuracy: ', accuracy)
        # Feature importance
        st.subheader('Feature Importance')
        importance = pd.DataFrame({'feature': X_train.columns, 'importance': clf.feature_importances_})
        importance = importance.sort_values('importance', ascending=False)
        st.bar_chart(importance.set_index('feature'))

# Line plot of user's input
        st.subheader('Your Input')
        user_input = pd.DataFrame([[age, bmi, nmisc, activity, location, temp, bpm, stress, bp]], columns=X.columns)
        st.line_chart(user_input.T)
# Generate report
report = f"""
Health Report and Prescription
------------------------------
Age: {age}
BMI: {bmi}
Number of previous miscarriages: {nmisc}
Activity level: {activity}
Location: {location}
Body temperature: {temp}
Heart rate: {bpm}
Stress level: {stress}
Blood pressure level: {bp}

Accuracy: {accuracy}

Prescription:
- Maintain a healthy diet rich in fruits, vegetables, lean proteins, and whole grains.
- Regular physical activity can help maintain a healthy weight and overall health.
- Avoid smoking and limit alcohol consumption.
- Regular check-ups with your healthcare provider can help detect any potential health issues early.
- Mental health is just as important as physical health. Consider activities like meditation or yoga to reduce stress.
"""

# Download report
st.download_button(
    label="Download Health Report and Prescription",
    data=report,
    file_name="health_report.txt",
    mime="text/plain"
)