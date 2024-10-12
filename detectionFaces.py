import os
import pandas as pd
import cv2
from deepface import DeepFace

data = {
    "Name":[],
    "Age":[],
    "Gender":[],
    "Race":[]
}

for file in os.listdir("faces"):
    img = cv2.imread(f"faces/{file}")

    results = DeepFace.analyze(img,actions=["age","gender","race"])

    result = results[0]

    data["Name"].append(file.split(".")[0])
    data["Age"].append(result["age"])
    data["Gender"].append(result["dominant_gender"])
    data["Race"].append(result["dominant_race"])

df = pd.DataFrame(data)
print(df)

df.to_csv("people.csv",index=False)


