import streamlit as st
import pandas as pd
import joblib

# Load saved model + columns
model = joblib.load("student_model.pkl")
model_cols = pd.read_json("model_columns.json", typ="series")

st.title("Student Performance Prediction (H, M, L)")

# Form inputs
gender = st.selectbox("Gender", ["M", "F"])
nationality = st.text_input("Nationality (e.g., KW)")
place = st.text_input("Place of Birth (e.g., Kuwait)")

stage = st.selectbox("Stage ID", ["lowerlevel", "MiddleSchool", "HighSchool"])
grade = st.text_input("Grade ID (e.g., G-07)")
section = st.selectbox("Section", ["A", "B", "C"])
topic = st.selectbox("Topic", ["IT", "Math", "English", "Science"])

semester = st.selectbox("Semester", ["F","S"])
relation = st.selectbox("Relation", ["Father","Mum"])

raisedhands = st.number_input("Raised Hands", 0, 200)
vis = st.number_input("Visited Resources", 0, 300)
ann = st.number_input("Announcements View", 0, 300)
dis = st.number_input("Discussion", 0, 300)

survey = st.selectbox("Parent Answering Survey", ["Yes", "No"])
satisfaction = st.selectbox("Parent School Satisfaction", ["Good", "Bad"])
absence = st.selectbox("Absence Days", ["Under-7", "Above-7"])

# Button to predict
if st.button("Predict Performance"):
    raw = {
        "gender": gender,
        "NationalITy": nationality,
        "PlaceofBirth": place,
        "StageID": stage,
        "GradeID": grade,
        "SectionID": section,
        "Topic": topic,
        "Semester": semester,
        "Relation": relation,
        "raisedhands": raisedhands,
        "VisITedResources": vis,
        "AnnouncementsView": ann,
        "Discussion": dis,
        "ParentAnsweringSurvey": survey,
        "ParentschoolSatisfaction": satisfaction,
        "StudentAbsenceDays": absence
    }
    
    df = pd.DataFrame([raw])
    df_enc = pd.get_dummies(df, drop_first=False).reindex(columns=model_cols, fill_value=0)
    pred = model.predict(df_enc)[0]

    st.success(f"🎯 Predicted Class: **{pred}**")