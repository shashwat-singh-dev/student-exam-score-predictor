from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware
'''
What is CORS?
Your frontend is running here:
http://127.0.0.1:5500

Your FastAPI backend is running here:
http://127.0.0.1:8000

Browser sees these as different origins because the ports are different.
Frontend
127.0.0.1:5500
       ↓
       ❌ Browser blocks request
       ↓
Backend
127.0.0.1:8000

This is a browser security mechanism called CORS (Cross-Origin Resource Sharing).
So we need to tell FastAPI:

"I allow my frontend at port 5500 to communicate with me."

'''
# FastAPI mein user ke input ko define karne ke liye Pydantic use karenge.

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins= ["http://127.0.0.1:5500","http://127.0.0.1:8000"],
    allow_credentials= True,
    allow_methods= ["*"],
    allow_headers=["*"]
)

# Load saved objects
encoder = joblib.load("encoder.pkl")
linear_model = joblib.load("linear_regression_model.pkl")
ridge_model = joblib.load("ridge_model.pkl")
lasso_model = joblib.load("lasso_model.pkl")


class StudentInput(BaseModel):
    Hours_Studied: float
    Attendance: float
    Sleep_Hours: float
    Previous_Scores: float
    Tutoring_Sessions: float
    Physical_Activity: float

    Parental_Involvement: str
    Access_to_Resources: str
    Extracurricular_Activities: str
    Motivation_Level: str
    Internet_Access: str
    Family_Income: str
    Teacher_Quality: str
    School_Type: str
    Peer_Influence: str
    Learning_Disabilities: str
    Parental_Education_Level: str
    Distance_from_Home: str
    Gender: str


@app.post("/predict")
def predict(data: StudentInput):

    input_data = pd.DataFrame([data.model_dump()])

    numerical_cols = [
        "Hours_Studied",
        "Attendance",
        "Sleep_Hours",
        "Previous_Scores",
        "Tutoring_Sessions",
        "Physical_Activity"
    ]

    categorical_cols = [
        "Parental_Involvement",
        "Access_to_Resources",
        "Extracurricular_Activities",
        "Motivation_Level",
        "Internet_Access",
        "Family_Income",
        "Teacher_Quality",
        "School_Type",
        "Peer_Influence",
        "Learning_Disabilities",
        "Parental_Education_Level",
        "Distance_from_Home",
        "Gender"
    ]

    encoded_data = encoder.transform(input_data[categorical_cols])

    numerical_data = input_data[numerical_cols].values

    final_input = pd.concat(
        [
            pd.DataFrame(numerical_data),
            pd.DataFrame(encoded_data.toarray())
        ],
        axis=1
    )

    linear_prediction = linear_model.predict(final_input)[0]
    ridge_prediction = ridge_model.predict(final_input)[0]
    lasso_prediction = lasso_model.predict(final_input)[0]

    return {
        "Linear Regression": round(float(linear_prediction), 2),
        "Ridge Regression": round(float(ridge_prediction), 2),
        "Lasso Regression": round(float(lasso_prediction), 2)
    }