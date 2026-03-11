from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np


def evaluate_model(trainer, dataset):

    preds = trainer.predict(dataset)

    y_pred = np.argmax(preds.predictions, axis=1)

    y_true = preds.label_ids

    acc = accuracy_score(y_true, y_pred)

    print("Accuracy:", acc)

    print(classification_report(y_true, y_pred))