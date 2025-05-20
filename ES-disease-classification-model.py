import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


'''
The disorders in this group are 
- psoriasis (1)
- seborrheic dermatitis (2)
- lichen planus (3)
- pityriasis rosea (4)
- chronic dermatitis (5)
- pityriasis rubra pilaris (6).

Usually, a biopsy is necessary for the diagnosis, but unfortunately, these diseases 
share many histopathological features as well (columns 12-33).

Link to more information on the data: 
https://www.kaggle.com/datasets/olcaybolat1/dermatology-dataset-classification/data
'''

# 1 - Cleaning dataset and analyzing data

df = pd.read_csv("./dermatology_database_1.csv")
df.head()
invalid_ages = df.loc[df["age"] == "?"].index
df.drop(index=invalid_ages, inplace=True)
df.reset_index(inplace=True)

# 2 - Creating the model

y = df["class"]
X = df.drop(columns="class")
model = DecisionTreeClassifier()

for split_size in [0.05, 0.1, 0.2, 0.4, 0.8]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split_size, random_state=1)
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"Accuracy with train_size={split_size}: {test_accuracy}")


#3 - Finding opportunities for improvement

'''Note: this section is running the model using train_size = 0.8'''

train_scores = model.predict_proba(X_train)

importances = model.feature_importances_
feature_names = model.feature_names_in_
sorted_indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.title("Feature Importances")
plt.bar(range(len(feature_names)), importances[sorted_indices], align='center')
plt.xticks(range(len(feature_names)), np.array(feature_names)[sorted_indices], rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

print(classification_report(y_test, test_pred))