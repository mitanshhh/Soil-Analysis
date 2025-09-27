# 🌱 Soil Analysis


A Python project that predicts soil fertility and provides AI-driven improvement recommendations for better crop yield.  
It combines **Random Forest classification** with AI-based nutrient analysis to suggest actionable soil improvement steps.

Live Project link: https://huggingface.co/spaces/pymit/Soil-Analysis

---

## **Features**

- Predict soil fertility using **Random Forest** based on N, P, K, and pH values.
- AI-powered suggestions for improving soil nutrients (Nitrogen, Potassium,Phosphorus and pH).
- Automatically calculates whether each nutrient is **ample or deficient**.
- Provides **real-world solutions**: fertilizer type, compost, or soil amendments with suggested amounts.
- Integration with **MLflow** for experiment tracking and logging metrics.
- Streamlit frontend for **interactive input and visualization**.

---

## **Installation**

1. Clone the repository:
```bash
git clone https://github.com/mitanshhh/Soil-Analysis.git
```

2 Create and activate a virtual environment:

```
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

3 Install required dependencies:

```
pip install -r requirements.txt
```


4 Create a .env file and add your API key for [ChatGroq](https://console.groq.com/keys):

```
GROQ_API_KEY=your_api_key_here
```


## Usage

Run the Streamlit app:

```
streamlit run main.py
```
---

Enter Nitrogen, Phosphorus, Potassium, and pH values for your soil.

Click Predict Fertility.

View the predicted fertility status.

Check AI-driven improvement recommendations for each nutrient.

Example Output
```
[
  ["Nitrogen", "deficient", "Apply urea at 100 kg/ha or add nitrogen-rich compost such as cow dung manure..."],
  ["Potassium", "deficient", "Incorporate potassium sulfate at 75 kg/ha or potassium-rich organic amendments..."],
  ["ph", "ample", "Use elemental sulfur or acidifying organic matter to lower pH by approximately 3.2 units..."]
]
```



## Project Structure
```text
Soil-Analysis/
├── ML/
│   └── soil_analysis.pkl          # Trained Random Forest model
|   └── ml_model.py                # Python Code for RFC
├── Dataset/
│   └── soil_data.csv              # Soil dataset
|   └── Crop_recommendation.csv    # Actual dataset used for analysis and data manipulation
├── main.py                        # Streamlit frontend + backend integration
├── data_analysis.py               # Python code used for data analysis
├── .env                           # Environment variables (GROQ_API_KEY)
├── README.md                      # Project documentation
└── requirements.txt               # Project dependencies
```

## Keypoint 
MLflow Integration

Tracks precision, recall, and F1-score for your Random Forest model.

Logs model versioning and metrics for reproducibility.

## License: "MIT License"

## Acknowledgements:
    - Streamlit for interactive UI
    - LangChain for AI integration
    - MLflow for experiment tracking
