# 🩺 Heart Disease Prediction Pipeline 🚀

This project is a comprehensive, end-to-end machine learning pipeline designed to predict the presence of heart disease. Starting with raw, messy data, we walk through every essential step: data cleaning, exploratory data analysis (EDA), feature selection, model training, and hyperparameter tuning. The final result is a fully optimized predictive model deployed in an interactive web application built with Streamlit.

---

## ✨ Key Features

*   **Robust Data Cleaning:** Handles missing values, inconsistent formats, and even unique data entry errors (like numbers spelled out as words).
*   **In-Depth EDA:** Generates visualizations like histograms and correlation heatmaps to understand the data's story.
*   **Intelligent Feature Selection:** Uses three distinct methods (Random Forest Importance, RFE, Chi-Square) to identify the most predictive medical features.
*   **Model Bake-Off:** Trains and evaluates four different supervised learning models (Logistic Regression, Decision Tree, Random Forest, SVM) to find the most promising candidate.
*   **Performance Optimization:** Fine-tunes the champion model using `GridSearchCV` to squeeze out the best possible performance.
*   **Interactive Web UI:** A simple and intuitive web application built with **Streamlit** that allows users to input patient data and get a real-time risk prediction.
*   **Ready for Deployment:** Includes instructions for sharing the local web app online using **Ngrok**.

---

## ⚙️ Project Workflow

The project follows a classic machine learning pipeline structure:
[Raw Data] -> [1. Data Cleaning & Preprocessing] -> [2. Exploratory Data Analysis] -> [3. Feature Selection] -> [4. Model Training] -> [5. Hyperparameter Tuning] -> [6. Saved Champion Model] -> [7. Streamlit Web App]

---

## 🛠️ Technologies Used

*   **Data Science & Machine Learning:** Pandas, NumPy, Scikit-learn
*   **Data Visualization:** Matplotlib, Seaborn
*   **Web Application:** Streamlit
*   **Development Environment:** Jupyter Notebooks
*   **Model Persistence:** Joblib

---

## 🚀 Getting Started: Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Prerequisites
*   Python 3.8 or higher installed.
*   `git` installed for cloning the repository.

### 2. Installation Steps

**1. Clone the repository:**
```bash
git clone https://github.com/your-username/Heart-Disease-Project.git
cd Heart-Disease-Project
```

**2. Create and activate a virtual environment:**
This keeps the project's dependencies isolated.
```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.env\Scriptsctivate
```

**3. Install the required libraries:**
```bash
pip install -r requirements.txt
```

**4. Download the Dataset:**
This project uses the "Heart Disease UCI" dataset.  
Important: You must download the data file yourself. It is often named processed.cleveland.data or similar.  
Place the downloaded file inside the `/data/` directory.  
Rename the file to `heart_disease.csv`. The code is configured to look for this specific name.

---

## 🏃‍♀️ How to Run the Project

There are two main parts to this project: the analysis notebooks and the final web application.

### 1. Running the Jupyter Notebooks
The notebooks in the `/notebooks/` directory walk through the entire analysis. It's essential to run them in order, as they generate the files needed for the next steps (like the cleaned dataset and the final model).

Start Jupyter Notebook from your terminal:
```bash
jupyter notebook
```

Open and run the notebooks in numerical order:
* `01_data_preprocessing.ipynb`
* `02_pca_analysis.ipynb`
* ...and so on, up to `06_hyperparameter_tuning.ipynb`.

After running notebook 06, you will have the final, trained model (`final_model.pkl`) and the scaler (`scaler.pkl`) saved in the `/models/` directory.
