import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)
df=pd.read_csv("loan_data.csv")


pre_df = pd.get_dummies(df,columns=['purpose'],drop_first=True)
display(pre_df.head())

# X = pre_df.drop('not.fully.paid', axis=1)
# y = pre_df['not.fully.paid']

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.33, random_state=125
# )

# model = GaussianNB()

# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)

# accuray = accuracy_score(y_pred, y_test)
# f1 = f1_score(y_pred, y_test, average="weighted")

# print("Accuracy:", accuray)
# print("F1 Score:", f1)