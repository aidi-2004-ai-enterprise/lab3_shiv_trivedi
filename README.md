Lab 3 – Penguins Classification with XGBoost and FastAPI

🐧 Overview
This project implements a machine learning pipeline using the Seaborn Penguins dataset. It includes:
- Preprocessing using one-hot and label encoding
- Model training using XGBoost
- Prediction API served using FastAPI
- Robust input validation with Pydantic
- Logging and error handling
- Screen demo of both successful and failed predictions

📁 Project Structure
lab3_shiv_trivedi/
├── train.py
├── app/
│   ├── main.py
│   ├── data/
│   │   ├── model.json
│   │   ├── label_map.json
├── requirements.txt
├── README.md
├── demo.mp4

⚙️ Setup Instructions

1. Create and Activate Virtual Environment
python -m venv .venv
.venv\Scripts\activate

2. Install Required Packages
pip install -r requirements.txt

🚀 Run the Application

Train the model:
python train.py

Run the FastAPI server:
uvicorn app.main:app --reload

Then open:
http://127.0.0.1:8000/docs

📬 Sample Request (Swagger)
{
"bill_length_mm": 45.2,
"bill_depth_mm": 14.8,
"flipper_length_mm": 210,
"body_mass_g": 5000,
"year": 2008,
"sex": "male",
"island": "Biscoe"
}

🧪 Features
- ✅ Pydantic validation (sex and island use enums)
- ✅ Graceful failure for invalid values
- ✅ Manual one-hot encoding to match training columns
- ✅ Logging for request and prediction flow
- ✅ Model evaluation printed in terminal during training

📹 Demo Video
The video file demo.mp4 includes:
- A successful prediction via Swagger UI
- A failed prediction with invalid sex or island values
- Running the Uvicorn server from terminal

👤 Author
Shivkumar Trivedi
Durham College – AIDI 2004
July 2025
