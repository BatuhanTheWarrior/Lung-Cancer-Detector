import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("lung_cancer.csv")

X = df[['AGE', 'SMOKING', 'ALCOHOL', 'CHRONIC_DISEASE', 'FATIGUE']]
y = df['LUNG_CANCER']

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open("lung_model.pkl", "wb"))

print("Model trained and saved!")