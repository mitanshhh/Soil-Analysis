import streamlit as st
import joblib
import pandas as pd
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage,SystemMessage
import os 


load_dotenv()


rf = joblib.load(r"ML\soil_analysis.pkl")
df = pd.read_csv(r"Dataset\soil_data.csv")
df_good_soil = df[df['Fertility_Status'] == 'Good'][['N', 'P', 'K', 'ph']].mean()



st.title("🌱 Soil Fertility Predictor")
st.subheader("Enter Soil Nutrient Values:")

N = st.number_input("Nitrogen (kg/ha)", min_value=10, max_value=145)
P = st.number_input("Phosphorus (kg/ha)", min_value=10, max_value=145)
K = st.number_input("Potassium (kg/ha)", min_value=10, max_value=200)
ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0)



def calc_improvement(N, P, K, ph):
        improvements = {'Nitrogen': '', 'Phosphorus': '', 'Potassium': '', 'ph': ''}
        if df_good_soil["N"] > N:
            improvements['Nitrogen'] = str(df_good_soil["N"] - N) + " deficient"
        else:
            improvements['Nitrogen'] = str(N - df_good_soil["N"]) + " ample"
        if 25 < P:
            improvements['Phosphorus'] = 'Soils are P-rich, no Phosphorus fertilizer needed. Stop P fertilization, use gypsum or lime to improve soil structure, and grow P-absorbing crops or cover crops to draw down excess P.'
        elif 15 <= P <= 25:
            improvements['Phosphorus'] = 'Adequate for most crops, apply maintenance doses'
        else:
            improvements['Phosphorus'] = 'Deficient, Apply a phosphorus-rich fertilizer (e.g., bone meal, rock phosphate) or organic amendments and ensure proper pH for uptake.'
        if df_good_soil["K"] > K:
            improvements['Potassium'] = str(df_good_soil["K"] - K) + " deficient"
        else:
            improvements['Potassium'] = str(K - df_good_soil["K"]) + " ample"
        if df_good_soil["ph"] > ph:
            improvements['ph'] = str(df_good_soil["ph"] - ph) + " deficient"
        else:
            improvements['ph'] = str(ph - df_good_soil['ph']) + " ample"

        return improvements


def ask_improvement_ai_model():
    improvement_req_api = f"""

Youve been given a soil dataset which has Nitrogen Potassium Phosphorus and pH level in it
Just in front of that its mentioned that the given element present in soil is deficient or ample
If its written ample then its present more in soil than ideal range 
If its written deficient then its present less than the ideal range
RETURN JSON OBJECT

Nitrogen Phosphorus Potassium and pH This is the ideal conditional range for good soil 
"Nitrogen": 40 - 60, 
"Phosphorus": 15 - 30, 
"Potassium": 100 - 150, 
"pH": 6-7 
Refer this range for ideal soil DO NOT HALLUCINATE FOR READINGS and RETURN JSON OBJECT
READ ELEMENT INPUT AND MAKE SURE IT MATCHES WITH YOUR OUTPUT 
ex - Input is ample and youre saying low this should not happen


You first compare these values with the ideal range values of N P K pH in a good fertilized soil 
Then give solution on what needs to be done to improve the fertility of soil in case of every element tell what things to add or remove from soil like any fertilizer or compost
Mention by how much amount the compound needs to be added or removed by proper calculation (kg/ha)

Return a JSON file like
    (
        "Nitrogen": (
            "status": "Low",
            "solution": "Apply urea at 100 kg/ha or add nitrogen-rich compost such as cow dung manure."
        )
    ),

IMPORTANT:
Return only valid JSON Object
Do not include explanations, markdown, or text outside the JSON.
GIVE "solution" all together as a string not as 'action': 'Add', 'amount': '29.28108108108108 kg/ha (29.28 kg/ha)', 'compound': 'potassium sulphate'
Solution should be of minimum 50 words with accurate numbers of what compound to use or what remedies to take
IN STATUS ITS MENTIONED IF ITS HIGH OR LOW IN INPUT SO TREAT IT ACCORDINGLY


Soil Data:
{improvement_needed}

"""
    messages = [
    SystemMessage(content="You are a JSON API. Always respond with ONLY valid JSON. No text, no explanations, no code fences."),
    HumanMessage(content=improvement_req_api)]

    llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")
    try:
        response = llm.invoke(messages)
        ai_model_response_json = json.loads(response.content)
        # print(ai_model_response_json)
        status_solution_list = [
            [element, val.get("status", "No status provided"), val.get("solution", "No solution provided")]
            for element, val in ai_model_response_json.items()
            # if element != "Phosphorus"  
        ]
    except Exception as api_error:
        print(api_error)
        print("Check API key or try again later")
    return status_solution_list

#Frontend

if st.button("Predict Fertility"):
    user_input = pd.DataFrame({'N':[N], 'P':[P], 'K':[K], 'ph':[ph]})

    predicted_data = rf.predict(user_input)[0]
    st.success(f"🌿 Predicted Fertility Status: {predicted_data}")
    

    improvement_needed = calc_improvement(N, P, K, ph)
    phosphorus_solution = improvement_needed['Phosphorus']

    with st.spinner("Checking for Improvements", show_time=True):
        result_list = ask_improvement_ai_model()
        
        
        st.subheader("💡 Soil Improvement Recommendations:")
        
        for i in range(0,4):
            st.write(f"## {result_list[i][0]}: {result_list[i][1]}")
            st.write(f'Solution: {result_list[i][2]}')
            



