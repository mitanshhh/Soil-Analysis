import pandas as pd
import numpy as np

df = pd.read_csv(r"Dataset\Crop_recommendation.csv")
df = df[["N","P","K","ph"]]

def fertility_status(row):
    score = 0
    
    # Nitrogen
    if row['N'] < 40:
        score += 0
    elif 40 <= row['N'] <= 70:
        score += 1
    else:
        score += 2

    # Phosphorus
    if row['P'] < 25:
        score += 0
    elif 30 <= row['P'] <= 60:
        score += 1
    else:
        score += 2

    # Potassium
    if row['K'] < 85:
        score += 0
    elif 90 <= row['K'] <= 150:
        score += 1
    else:
        score += 2

    # pH
    if 5.5 <= row['ph'] <= 7.5:
        score += 2
    elif 5.0 <= row['ph'] < 5.5 or 7.5 < row['ph'] <= 8.0:
        score += 1
    else:
        score += 0

    # Classify fertility
    if score >= 6:
        return "Good"
    elif score >= 4:
        return "Moderate"
    else:
        return "Poor"
    
df['Fertility_Status'] = df.apply(fertility_status, axis=1)
counts = df['Fertility_Status'].value_counts()
print(counts)
df.to_csv("Dataset\soil_data.csv")