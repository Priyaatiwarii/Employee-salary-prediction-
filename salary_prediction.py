import pandas as pd
import xgboost as xgb
import streamlit as st

def main():

    st.title("Employee Salary Prediction")

    st.write(
        "This app will help you predict employee salary."
    )

    # Load Model
    model = xgb.XGBRegressor()
    model.load_model("salary_model.json")

    # Inputs
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=25
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    gender = 1 if gender == "Male" else 0

    education = st.selectbox(
        "Education",
        ["Graduate", "Post Graduate", "PhD"]
    )

    # Encode Education
    education_dict = {
        "Graduate": 0,
        "Post Graduate": 1,
        "PhD": 2
    }

    education = education_dict[education]

    experience = st.number_input(
        "Experience (Years)",
        min_value=0,
        max_value=40,
        value=5
    )

    department = st.selectbox(
        "Department",
        ["IT", "HR", "Finance", "Sales"]
    )

    # Encode Department
    dept_dict = {
        "IT": 0,
        "HR": 1,
        "Finance": 2,
        "Sales": 3
    }

    department = dept_dict[department]

    job_level = st.number_input(
        "Job Level",
        min_value=1,
        max_value=10,
        value=3
    )

    performance_rating = st.slider(
        "Performance Rating",
        1,
        5,
        4
    )

    certifications = st.number_input(
        "Certifications",
        min_value=0,
        max_value=20,
        value=5
    )

    overtime_hours = st.number_input(
        "Overtime Hours",
        min_value=0,
        max_value=50,
        value=12
    )

    remote_work = st.selectbox(
        "Remote Work",
        ["Yes", "No"]
    )

    remote_work = 1 if remote_work == "Yes" else 0

    city = st.selectbox(
        "City",
        ["Delhi", "Mumbai", "Bangalore", "Chennai"]
    )

    # Encode City
    city_dict = {
        "Delhi": 0,
        "Mumbai": 1,
        "Bangalore": 2,
        "Chennai": 3
    }

    city = city_dict[city]

    company_tenure = st.number_input(
        "Company Tenure",
        min_value=0,
        max_value=40,
        value=6
    )

    projects_completed = st.number_input(
        "Projects Completed",
        min_value=0,
        max_value=100,
        value=15
    )

    skill_score = st.slider(
        "Skill Score",
        0,
        100,
        88
    )

    # Create DataFrame with EXACTLY 14 Features
    data_new = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [gender],
            "Education": [education],
            "Experience_Years": [experience],
            "Department": [department],
            "Job_Level": [job_level],
            "Performance_Rating": [performance_rating],
            "Certifications": [certifications],
            "Overtime_Hours": [overtime_hours],
            "Remote_Work": [remote_work],
            "City": [city],
            "Company_Tenure": [company_tenure],
            "Projects_Completed": [projects_completed],
            "Skill_Score": [skill_score]
        }
    )

    # Prediction
    if st.button("Predict Salary"):

        prediction = model.predict(data_new)

        st.success(
            f"Predicted Salary: ₹ {prediction[0]:,.2f} LPA"
        )


if __name__ == "__main__":
    main()
