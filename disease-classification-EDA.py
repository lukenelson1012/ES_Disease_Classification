import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./dermatology_database_1.csv")
df.head()
invalid_ages = df.loc[df["age"] == "?"].index
df.drop(index=invalid_ages, inplace=True)
df.reset_index(inplace=True)

conditions = ["psoriasis (1)", "sebhorric dermatitis (2)", "lichen planus (3)", "pityriasis rosea (4)", "chronic dermatitis (5)", "pityriasis rubra pilaris (6)"]

# 1 - Finding median age by age

df["age"] = df["age"].astype("int64")

plt.figure(figsize = (10, 6))
plt.bar(df["class"].unique(), df.groupby("class")["age"].mean())
plt.title("Average age for each sub-category of Eryhemato-Sqaumous Disease")
plt.xticks(range(1, len(conditions)+1), conditions, rotation=45)
plt.xlabel("Classification")
plt.ylabel("Average age")
plt.tight_layout()
plt.show()

# 2 - Distribution of itching severity per class

fig, ax = plt.subplots(3, 2, figsize=(10,40))
itching = []

for sev in range(0, 4):
    class_per_itching_severity = []
    for cls in [1, 2, 3, 4, 5, 6]:
        class_per_itching_severity.append(len(df.loc[(df["class"] == cls) & (df["itching"] == sev), "itching"].index))
    itching.append(class_per_itching_severity)

colors = ["red", "orange", "yellow", "green", "blue", "purple"]

for cls in range(0, 6):
    itching_severity_per_class = []
    for sev in range(0, 4):
        itching_severity_per_class.append(itching[sev][cls])
    
    ax[int(cls/2), cls%2].bar([0, 1, 2, 3], itching_severity_per_class, color=colors[cls])
    ax[int(cls/2), cls%2].set_title(conditions[cls])

    if cls % 2 == 0:
        ax[int(cls/2), cls%2].set_ylabel("Count")
    if cls / 2 >= 2:
        ax[int(cls/2), cls%2].set_xlabel("Severity")
        ax[int(cls/2), cls%2].set_xticks(range(0, 4), ["0", "1", "2", "3"])

plt.show()