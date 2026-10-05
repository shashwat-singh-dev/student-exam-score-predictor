<div align="center">

# Student Exam Score Predictor

**An end-to-end machine learning regression application that predicts a student's exam score from academic, behavioral, and background factors.**

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)

</div>

---

## Overview

Student Exam Score Predictor is a regression application that estimates a student's `Exam_Score` from 19 input features covering study habits, attendance, prior performance, and family and school context.

A user enters student information through a web form. The frontend sends the data to a **FastAPI** backend, which validates the request, applies a previously fitted `OneHotEncoder`, and returns predictions from three trained regression models: **Linear Regression**, **Ridge Regression**, and **Lasso Regression**.

The application currently runs locally for development. It is not deployed publicly.

## Why This Project

The goal was not only to train a regression model, but to build the full workflow around it and understand how each stage connects to the next:

```
Dataset → Data Cleaning → Missing Value Handling → Feature Separation
→ Train/Test Split → Categorical Encoding → Model Training
→ Model Evaluation → Model Serialization → FastAPI Backend
→ Frontend Integration → Prediction
```

The project was built from scratch for learning and portfolio development.

## Features

- Multi-model comparison: Linear, Ridge, and Lasso regression
- Missing categorical values preserved as an explicit `"Unknown"` category
- Leakage-aware preprocessing: the encoder is fitted on training data only
- FastAPI backend with Pydantic request validation
- Serialized model artifacts loaded at inference time
- Custom HTML/CSS/JavaScript frontend with a responsive layout
- CORS configured for local frontend–backend development
- Interactive API documentation via Swagger UI

## Machine Learning Workflow

| Stage | Description |
|---|---|
| **1. Data loading** | Load `StudentPerformanceFactors.csv` (6,607 rows, 20 columns) |
| **2. EDA** | Inspect distributions and identify missing values in three categorical columns |
| **3. Cleaning** | Replace missing categorical values with `"Unknown"` |
| **4. Feature separation** | Split inputs into 6 numerical and 13 categorical features |
| **5. Train/test split** | 80/20 split with `random_state=42` |
| **6. Encoding** | `OneHotEncoder(handle_unknown="ignore")`, fitted on `X_train` only |
| **7. Training** | Fit Linear, Ridge, and Lasso regression models |
| **8. Evaluation** | Compare MAE, RMSE, and R² on the held-out test set |
| **9. Serialization** | Save the encoder and models with `joblib` |
| **10. Serving** | Expose predictions through a FastAPI endpoint consumed by the frontend |

## Dataset

**Source file:** `StudentPerformanceFactors.csv`

| Property | Value |
|---|---|
| Rows | 6,607 |
| Columns | 20 |
| Input features | 19 |
| Target variable | `Exam_Score` |

**Numerical features (6)**

`Hours_Studied` · `Attendance` · `Sleep_Hours` · `Previous_Scores` · `Tutoring_Sessions` · `Physical_Activity`

**Categorical features (13)**

`Parental_Involvement` · `Access_to_Resources` · `Extracurricular_Activities` · `Motivation_Level` · `Internet_Access` · `Family_Income` · `Teacher_Quality` · `School_Type` · `Peer_Influence` · `Learning_Disabilities` · `Parental_Education_Level` · `Distance_from_Home` · `Gender`

## Data Preprocessing

### Missing values

EDA found missing values in three categorical columns:

| Column | Missing values | Share |
|---|---:|---:|
| `Teacher_Quality` | 78 | ~1.18% |
| `Parental_Education_Level` | 90 | ~1.36% |
| `Distance_from_Home` | 67 | ~1.01% |

Rather than imputing the mode, missing values were replaced with the explicit category **`"Unknown"`**. Mode imputation would assume the missing value belongs to the most common category. Using `"Unknown"` keeps the fact that the value was missing. After this step, no missing values remained in these columns.

### Encoding and leakage prevention

Categorical variables were converted with:

```python
OneHotEncoder(handle_unknown="ignore")
```

The encoder was **fitted only on `X_train`**, then used to transform both `X_train` and `X_test`. This prevents information from the test set leaking into preprocessing. The same saved encoder is reused at inference time.

```
6 numerical features + 37 one-hot encoded features = 43 model input features
```

### Train/test split

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

| Split | Samples |
|---|---:|
| Training | 5,285 |
| Testing | 1,322 |

## Model Development

Three regression models were trained on the same 43-feature representation. The aim was to establish a simple baseline and compare it with regularized linear approaches.

| Model | Configuration |
|---|---|
| Linear Regression | Default settings (baseline) |
| Ridge Regression | `alpha = 1.0` |
| Lasso Regression | `alpha = 0.01` |

Trained artifacts were saved with `joblib`:

```
encoder.pkl
linear_regression_model.pkl
ridge_model.pkl
lasso_model.pkl
```

## Model Performance

Metrics were computed on the held-out test set (1,322 samples).

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.4499 | 1.8033 | 0.7699 |
| Ridge Regression (α = 1.0) | 0.4499 | 1.8033 | 0.7699 |
| Lasso Regression (α = 0.01) | 0.4567 | 1.8045 | 0.7696 |

*Ridge values at higher precision: MAE 0.4498833, RMSE 1.8033080, R² 0.7699397.*

**Interpretation**

All three models produced very similar results. Linear and Ridge Regression performed almost identically, so Ridge was not meaningfully better than the baseline. Lasso was slightly worse in this setup, but the gap is small. These results suggest that the relationships captured by this feature representation can already be modeled reasonably well by linear approaches.

Model selection should rest on validation performance and practical considerations, not on choosing a more complex model by default.

## Key Findings

1. **Missing values can carry information.** Missing categorical values do not always need mode imputation. Encoding them as `"Unknown"` preserved the missingness signal.
2. **One-hot encoding was necessary.** Several predictors were categorical and could not be used by the linear models directly.
3. **Fit preprocessing on training data only.** The encoder must be fitted on the training split and reused for test and inference data to avoid data leakage.
4. **Linear Regression is a strong baseline.** It performed well with this feature representation.
5. **Ridge matched Linear Regression.** Regularization produced almost no change in performance.
6. **Lasso was slightly worse here.** This was observed in this experiment and is not a general claim about Lasso.
7. **Deployment context adds value.** A model is more useful when integrated into an application than when it stays in a notebook.
8. **The stack connects.** The project linked model development, API development, and frontend integration into one workflow.

## System Architecture

```
┌──────────────────────────────┐
│             User             │
└──────────────┬───────────────┘
               │  enters student information
               ▼
┌──────────────────────────────┐
│    Frontend (HTML / CSS)     │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│          JavaScript          │
│   POST http://127.0.0.1:8000 │
│           /predict           │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│       FastAPI  /predict      │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Input Validation (Pydantic) │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│     Feature Preprocessing    │
│  (DataFrame, numeric / cat)  │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│    Saved OneHotEncoder       │
│        (encoder.pkl)         │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│        Trained Models        │
│  Linear · Ridge · Lasso      │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│     Predictions (JSON)       │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│    Frontend Results View     │
└──────────────────────────────┘
```

## API

### `POST /predict`

Accepts all 19 student features, validates them with Pydantic, and returns a prediction from each of the three trained models.

**Processing steps**

1. Validate the request body with Pydantic
2. Convert the request into a Pandas DataFrame
3. Separate numerical and categorical features
4. Apply the saved `OneHotEncoder`
5. Combine numerical and encoded categorical features
6. Generate predictions with all three trained models
7. Return the predictions as JSON

**Example request** (representative values)

```json
{
  "Hours_Studied": 20,
  "Attendance": 85,
  "Parental_Involvement": "Medium",
  "Access_to_Resources": "High",
  "Extracurricular_Activities": "Yes",
  "Sleep_Hours": 7,
  "Previous_Scores": 75,
  "Motivation_Level": "Medium",
  "Internet_Access": "Yes",
  "Tutoring_Sessions": 1,
  "Family_Income": "Medium",
  "Teacher_Quality": "Medium",
  "School_Type": "Public",
  "Peer_Influence": "Positive",
  "Physical_Activity": 3,
  "Learning_Disabilities": "No",
  "Parental_Education_Level": "College",
  "Distance_from_Home": "Near",
  "Gender": "Male"
}
```

**Example response** (illustrative only; actual values depend on the input and the trained models)

```json
{
  "linear_regression": 67.12,
  "ridge_regression": 67.12,
  "lasso_regression": 67.05
}
```

> The response above only shows the shape of the output. The values are not guaranteed predictions, and the exact field names depend on the implementation in `app.py`.

Interactive documentation is available at `http://127.0.0.1:8000/docs` while the backend is running.

## Frontend

The frontend is a custom HTML/CSS/JavaScript interface with no framework. It presents a structured form for student information, sends the data to `POST http://127.0.0.1:8000/predict`, and renders the returned predictions dynamically.

**Design characteristics**

- Serif (Times New Roman) typography
- Structured input sections with a clean, professional layout
- Responsive layout, including a single-column mobile view
- Refined form controls with hover, focus, and active states
- Subtle gradients and shadows
- Accessible focus states
- Reduced-motion support

**CORS:** the backend allows requests from the local development frontend.

| Component | Origin |
|---|---|
| Frontend (VS Code Live Server) | `http://127.0.0.1:5500` |
| Backend (Uvicorn) | `http://127.0.0.1:8000` |

## Project Structure

```
student-exam-score-predictor/
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── app.py                          # FastAPI application
├── encoder.pkl                     # Fitted OneHotEncoder
├── linear_regression_model.pkl
├── ridge_model.pkl
├── lasso_model.pkl
├── load_data.ipynb                 # EDA, preprocessing, training, evaluation
├── StudentPerformanceFactors.csv
└── .gitignore
```

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/shashwat-singh-dev/student-exam-score-predictor.git
cd student-exam-score-predictor
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
```

```bash
# Windows
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install fastapi uvicorn pandas joblib scikit-learn
```

## Running the Project

**Start the backend**

```bash
uvicorn app:app --reload
```

| Service | URL |
|---|---|
| API | `http://127.0.0.1:8000` |
| Swagger docs | `http://127.0.0.1:8000/docs` |

**Start the frontend**

1. Open the project in VS Code.
2. Install the **Live Server** extension if it is not already installed.
3. Right-click `frontend/index.html` and select **Open with Live Server**.
4. The app opens at `http://127.0.0.1:5500` and communicates with the local API.

The backend must be running for predictions to work.

## Example Prediction Flow

1. The user fills in the 19 student fields in the form and submits it.
2. `script.js` sends the values as JSON to `POST /predict`.
3. FastAPI validates the payload and builds a DataFrame.
4. Numerical and categorical columns are separated, and the categorical columns are transformed with the saved `OneHotEncoder`.
5. The combined 43-feature input is passed to the Linear, Ridge, and Lasso models.
6. The API returns three predictions as JSON.
7. The frontend displays the three results in the results panel.

## Technologies Used

| Category | Tools |
|---|---|
| Language | Python, JavaScript |
| Data & ML | Pandas, NumPy, Scikit-learn |
| Backend | FastAPI, Pydantic, Joblib |
| Frontend | HTML, CSS, JavaScript |
| Tooling | Jupyter Notebook, Git, GitHub |

## Future Improvements

These are planned directions. None are implemented yet.

- Build a complete scikit-learn `Pipeline` combining preprocessing and model
- Add cross-validation
- Add feature scaling where appropriate
- Tune regularization hyperparameters systematically
- Compare tree-based regression models
- Add a stronger validation methodology
- Add prediction confidence and uncertainty analysis
- Improve frontend visualization of results
- Containerize the application with Docker
- Deploy the FastAPI backend
- Deploy the frontend
- Add automated testing
- Add CI/CD
- Add model versioning

## Author

**Shashwat**
B.Tech student specializing in Artificial Intelligence and Machine Learning

GitHub: [shashwat-singh-dev](https://github.com/shashwat-singh-dev)

---

<div align="center">

*Built as an end-to-end learning and portfolio project.*

</div>
