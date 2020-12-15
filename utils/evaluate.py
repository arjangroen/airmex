from utils.model import rebuild_kneenet, predict
from utils.preprocess_image import preprocess
from sklearn.metrics import classification_report
import os
import cv2
import numpy as np

filepath_base = "data/mendeley/kneeKL299/test/"
filepaths = [filepath_base + str(kl) for kl in range(5)]

predictions = []
actuals = []
target_names = ["KL 0","KL 1", "KL 2", "KL 3", "KL 4"]

model = rebuild_kneenet()

kl=-1

for filepath_kl in filepaths:
    count = 0
    kl+=1
    for filename in os.listdir(filepath_kl):
        img = cv2.imread(os.path.join(filepath_kl, filename), 0).astype("float")
        processed_image = preprocess(img)
        logits, probabilities = predict(processed_image)
        predictions.append(np.argmax(probabilities))
        actuals.append(kl)
        count+=1

report = classification_report(actuals, predictions, target_names=target_names)

with open('testreport','w') as file:
    file.write(report)
