# 📉 Customer Churn Intelligence System

🚀 An AI-powered Customer Churn Prediction Dashboard designed to analyze customer behavior, predict churn probability, segment customers into risk levels, and recommend actionable retention strategies.

---

## 🔍 Project Overview

Customer churn is a major challenge for subscription-based businesses.  
This project builds a **complete end-to-end Machine Learning system + interactive dashboard** to:

- Predict customer churn probability  
- Segment customers into **Low, Medium, High, Critical risk**  
- Identify churn drivers (behavioral + business factors)  
- Generate **Top 50 high-risk customer watchlist**  
- Provide **data-driven retention strategies**  
- Visualize model performance (ROC, PR Curve, Feature Importance)

---

## 🎯 Key Features

### ✅ Customer Churn Prediction
- Predicts whether a customer will churn or not  
- Outputs probability score + risk level  

### ✅ Risk Segmentation
- Low Risk  
- Medium Risk  
- High Risk  
- Critical Risk  

### ✅ Business Insights
- Detects churn causes:
  - Payment delays  
  - Low usage / engagement  
  - High support tickets  
  - Contract type  
- Suggests targeted retention strategies  

### ✅ Model Performance Dashboard
Includes:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  
- PR-AUC  

📊 Visuals:
- Confusion Matrix  
- ROC Curve  
- Precision-Recall Curve  
- Feature Importance Graph  

### ✅ Top 50 Churn Watchlist
- Displays highest-risk customers  
- Helps prioritize retention efforts  

### ✅ Interactive Dashboard (Streamlit)
- Professional UI/UX  
- Sidebar navigation  
- Real-time predictions  

---

## 🧠 Tech Stack

- **Language:** Python  
- **ML Models:** Random Forest / XGBoost  
- **Libraries:**  
  - pandas, numpy  
  - scikit-learn  
  - imbalanced-learn (SMOTE)  
  - matplotlib, seaborn  
- **Frontend:** Streamlit  
- **Backend Ready:** FastAPI  
- **Model Saving:** joblib  

---

## 📂 Project Structure

Customer-Churn-Intelligence-System/
│
├── data/
│   └── churn_data.csv
│
├── models/
│   └── churn_model.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   ├── feature_importance.png
│   └── metrics.json
│
├── src/
│   ├── data_generator.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── prediction.py
│
├── app.py
├── main.py
├── requirements.txt
└── README.md

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
git clone https://github.com/YOUR_USERNAME/Customer-Churn-Intelligence-System.git  
cd Customer-Churn-Intelligence-System  

### 2️⃣ Create Virtual Environment
python -m venv venv  
source venv/bin/activate   (Mac/Linux)  

### 3️⃣ Install Dependencies
pip install -r requirements.txt  

---

## ▶️ Run Project

### Train Model + Generate Outputs
python main.py  

### Run Dashboard
streamlit run app.py  

---

## 💡 Business Impact

✔ Identify high-risk customers early  
✔ Reduce churn rate  
✔ Increase customer lifetime value  
✔ Improve marketing efficiency  
✔ Enable data-driven retention strategy  

---

## 📌 Use Cases

- Telecom companies  
- SaaS platforms  
- OTT services  
- Banking & FinTech  
- Subscription-based businesses  

---

## 🚀 Future Improvements

- Real-world dataset integration (IBM Telco)  
- Deep learning models  
- Real-time API deployment (FastAPI)  
- Database integration  
- Cloud deployment (AWS / GCP)  

---

## 👩‍💻 Author

**Ananya Pradhan**  
B.Tech IT | Data Science & AI Enthusiast  

GitHub:   https://github.com/ananya020pradhan
LinkedIn: www.linkedin.com/in/ananya-pradhan-10bb462ba

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to connect!
