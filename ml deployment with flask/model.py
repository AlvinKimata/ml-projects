import numpy as np
import pandas as pd

df = pd.read_csv('inputs/train.csv')
include = ['Age', 'Sex', 'Embarked', 'Survived'] # Only four features
df_ = df[include]

#Perform data processing.
categoricals = []
for col, col_type in df_.dtypes.iteritems():
    if col_type == 'O':
        categoricals.append(col)
    else:
        df_[col].fillna(0, inplace = True)

df_ohe = pd.get_dummies(df_, columns = categoricals, dummy_na=True)

#Logistic regression classifier.
from sklearn.linear_model import LogisticRegression

dependent_variable = 'Survived'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]

lr = LogisticRegression()
lr.fit(x, y)


#Save model.
import joblib
joblib.dump(lr, 'models/model.pkl')

print('Model dumped')

#Save data columns for training.
model_columns = list(x.columns)
joblib.dump(model_columns, 'models/model_columns.pkl')
print("Model columns dumped.")