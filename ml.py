import pandas as pd
from IPython.display import display
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    classification_report,
)
df = pd.read_csv("total_crimes.csv")
#display(df.head())
display(df['STATE/UT'].unique())
#display(df['STATE/UT'])SHOWS ALL ROWS ON STATE
print(len(df['STATE/UT'].unique()))
#df_new = df[df['STATE/UT'] == 'ANDHRA PRADESH']GIVES ROWS WITH ANDRA
#display(df_new)
#df_new1=df['STATE/UT'].values == 'ANDHRA PRADESH'
#display(df.loc[df_new1])
df_new2=df[(df['DISTRICT'] == 'ADILABAD')]
display(df_new2)
display(df['DISTRICT'])










#sns.countplot(data=df,x='STATE/UT',hue='MURDER')
#plt.show
def unique(list1):
    x = np.array(list1)
    print(np.unique(x))
pre_df = pd.get_dummies(df,columns=['STATE/UT'],drop_first=True)
display(pre_df)

X = pre_df.drop('MURDER', axis=1)
y = pre_df['MURDER']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
)

model = GaussianNB()

model.fit(X_train, y_train);

y_pred = model.predict(X_test)

accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuray)
print("F1 Score:", f1)
