import os
from deepface import DeepFace
import cv2
import pandas as pd

print("""
Welcome to a prediction model for age, gender and race of people in photos.
Created by modules such as Computer Vision 2, Deepface and Pandas, it can detect peoples emotions.
You CAN add images in the assets directory
An sample image set is available:
""")
for samples in os.listdir("assets"):
    print(samples)

confirm = input("\nContinue? [y/n] ")
if confirm.lower().strip() == "y":
    pass
elif confirm.lower().strip() == "n":
    raise Exception("ComfirmedInputInterrupt")
else:
    print("Bad input")

data = {
    "Name": [],
    "Gender": [],
    "Age": [],
    "Race": []
}

for img in os.listdir("assets"):
    final = DeepFace.analyze(cv2.imread(f"assets\{img}"), actions=("age", "gender", "race"))
    data["Name"].append(img.split(".")[0])
    data["Gender"].append(final[0]["dominant_gender"])
    data["Age"].append(final[0]["age"])
    data["Race"].append(final[0]["dominant_race"])

reprdata = pd.DataFrame(data)
print(reprdata)
reprdata.to_csv("results.csv")