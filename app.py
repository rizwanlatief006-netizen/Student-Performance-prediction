
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_performance_model.pkl")

# Website configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Prediction System")
st.write("Predict student academic performance using Machine Learning.")

st.info("Enter student details below to predict academic performance.")

# Student information form
with st.form("student_form"):

    st.subheader("Personal Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        school = st.selectbox("School", ["GP", "MS"])
        sex = st.selectbox("Gender", ["M", "F"])
        age = st.slider("Age", 15, 22, 17)
        address = st.selectbox("Address", ["U", "R"])
        famsize = st.selectbox("Family Size", ["LE3", "GT3"])

    with col2:
        Pstatus = st.selectbox("Parent Status", ["T", "A"])
        Medu = st.selectbox("Mother Education", [0, 1, 2, 3, 4])
        Fedu = st.selectbox("Father Education", [0, 1, 2, 3, 4])
        Mjob = st.selectbox("Mother Job", ["teacher", "health", "services", "at_home", "other"])
        Fjob = st.selectbox("Father Job", ["teacher", "health", "services", "at_home", "other"])

    with col3:
        reason = st.selectbox("Reason for Choosing School", ["home", "reputation", "course", "other"])
        guardian = st.selectbox("Guardian", ["mother", "father", "other"])
        traveltime = st.selectbox("Travel Time", [1, 2, 3, 4])
        studytime = st.selectbox("Weekly Study Time", [1, 2, 3, 4])
        failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])

    st.subheader("Academic and Lifestyle Information")

    col4, col5, col6 = st.columns(3)

    with col4:
        schoolsup = st.selectbox("Extra School Support", ["yes", "no"])
        famsup = st.selectbox("Family Support", ["yes", "no"])
        paid = st.selectbox("Extra Paid Classes", ["yes", "no"])
        activities = st.selectbox("Extra Activities", ["yes", "no"])
        nursery = st.selectbox("Attended Nursery", ["yes", "no"])

    with col5:
        higher = st.selectbox("Wants Higher Education", ["yes", "no"])
        internet = st.selectbox("Internet Access", ["yes", "no"])
        romantic = st.selectbox("In a Relationship", ["yes", "no"])
        famrel = st.slider("Family Relationship", 1, 5, 4)
        freetime = st.slider("Free Time", 1, 5, 3)

    with col6:
        goout = st.slider("Going Out", 1, 5, 3)
        Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1)
        Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1)
        health = st.slider("Health Status", 1, 5, 3)
        absences = st.number_input("School Absences", 0, 100, 2)

    submit = st.form_submit_button("Predict Student Performance")

if submit:

    student_data = {
        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,
        "Medu": Medu,
        "Fedu": Fedu,
        "Mjob": Mjob,
        "Fjob": Fjob,
        "reason": reason,
        "guardian": guardian,
        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,
        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,
        "higher": higher,
        "internet": internet,
        "romantic": romantic,
        "famrel": famrel,
        "freetime": freetime,
        "goout": goout,
        "Dalc": Dalc,
        "Walc": Walc,
        "health": health,
        "absences": absences
    }

    input_df = pd.DataFrame([student_data])

    prediction = model.predict(input_df)[0]
    st.write("DEBUG PREDICTION:",
             repr(prediction))
    st.write("MODEL CLASSES:",model.classes_)
    if hasattr(model, "predict_proba"):
        st.write("CLASS PROBABILITIES:",model.predict_proba(input_df)[0])

    st.subheader("Prediction Result")

    if prediction == "Good":
        st.success("Good Performance! 🌟")
    elif prediction == "Average":
        st.warning("Average Performance.")
    else:
        st.error("At Risk. Additional academic support may help.")

    st.write("Predicted Category:", prediction)

# Model comparison section
st.divider()
st.subheader("Machine Learning Algorithm Comparison")

try:
    results = pd.read_csv("model_comparison.csv")
    st.dataframe(results, use_container_width=True)

except Exception:
    st.write("Model comparison results are unavailable.")

st.caption("Academic mini project | Student Performance Prediction System")
st.caption("Predictions are estimates based on the training dataset.")
