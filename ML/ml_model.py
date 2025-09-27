import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,f1_score,recall_score,precision_score
import mlflow


df = pd.read_csv(r"Dataset\soil_data.csv")

X = df[['N','P','K','ph']]
y = df["Fertility_Status"]

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=40,test_size=0.1)

rf = RandomForestClassifier()
rf.fit(X_train,y_train)


y_pred = rf.predict(X_test)

# importances = rf.feature_importances_
# print(y_pred)

df_good_soil = df[df['Fertility_Status'] == 'Good']
print(df_good_soil['P'])
df_good_soil = df_good_soil[['N', 'P', 'K', 'ph']].mean()
print(df_good_soil['P'])


joblib.dump(rf,'soil_analysis.pkl')

print(classification_report(y_test,y_pred))

metrics = {
    'precision': precision_score(y_test,y_pred,average='macro'),
    'recall': recall_score(y_test,y_pred,average='macro'),
    "f1" : f1_score(y_test,y_pred,average='macro')
}

mlflow.set_experiment("Soil Analysis")
with mlflow.start_run():
    mlflow.set_tag('model1-rf','0.0.1')
    mlflow.sklearn.log_model(rf)
    mlflow.log_metrics(metrics)