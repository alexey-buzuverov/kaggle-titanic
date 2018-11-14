# Titanic Kaggle

import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load data
train_df = pd.read_csv('C:/Users/Alexey/Documents/Python/Kaggle-Titanik/input/train.csv', header=0)
test_df = pd.read_csv('C:/Users/Alexey/Documents/Python/Kaggle-Titanik/input/test.csv', header=0)

# Merge train and test sets
test_df.insert(1,'Survived',np.nan)
all_df = pd.concat([train_df, test_df])

# Perform corrections
corr_dict = {248: pd.Series([0,1], index=['SibSp', 'Parch'],),
             313: pd.Series([1,0], index=['SibSp', 'Parch'],),
             418: pd.Series([0,0], index=['SibSp', 'Parch'],),
             756: pd.Series([0,1], index=['SibSp', 'Parch'],),
             1041: pd.Series([1,0], index=['SibSp', 'Parch'],),
             1130: pd.Series([0,0], index=['SibSp', 'Parch'],),
             1170: pd.Series([2,0], index=['SibSp', 'Parch'],),
             1254: pd.Series([1,0], index=['SibSp', 'Parch'],)
             }

all_df[['SibSp','Parch']] = all_df.apply(lambda s: corr_dict[s['PassengerId']]
    if s['PassengerId'] in [248,313,418,756,1041,1130,1170,1254] else s[['SibSp','Parch']], axis = 1)

# Add Title
all_df.insert(3,'Title','')
all_df['Title'] =  all_df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# Replace rare titles
all_df.loc[all_df['Title'].isin(['Ms','Mlle']), 'Title'] = 'Miss'
all_df.loc[all_df['Title'].isin(['Mme','Lady','Dona','Countess']), 'Title'] = 'Mrs'
all_df.loc[all_df['Title'].isin(['Col','Major','Sir','Rev','Capt','Don','Jonkheer']), 'Title'] = 'Mr'
all_df.loc[(all_df['Title'] == 'Dr') & (all_df['Sex'] == 'male'),'Title'] = 'Mr'
all_df.loc[(all_df['Title'] == 'Dr') & (all_df['Sex'] == 'female'),'Title'] = 'Mrs'

# Fill Cabin
all_df['Cabin'].fillna('U',inplace=True)

# Add Deck
all_df.insert(11,'Deck','')
all_df['Deck'] =  all_df['Cabin'].map(lambda s: s[0])

# Add Family Size
all_df.insert(9,'FamSize','')
all_df['FamSize'] = all_df.apply(lambda s: 1+s['SibSp']+s['Parch'], axis = 1)

# Add Group Size
all_df.insert(10,'GrSize','')
ticket_counts = all_df['Ticket'].value_counts()
all_df['GrSize'] = all_df.apply(lambda s: ticket_counts.loc[s['Ticket']], axis=1)

# Add Familiy Name
all_df.insert(4,'Fname','')
all_df['Fname'] =  all_df.Name.str.extract('^(.+?),', expand=False)

# Add isAlone wSib wSp wCh wPar
all_df.insert(11,'isAlone',0)
all_df.insert(12,'wSib',0)
all_df.insert(13,'wSp',0)
all_df.insert(14,'wCh',0)
all_df.insert(15,'wPar',0)
# Fill new features

# Search for passengers with siblings
Pas_wSib = []
all_df_x_0 = all_df[(all_df['SibSp'] > 0) & (all_df['Parch'] == 0)]
name_counts_SibSp = all_df_x_0['Fname'].value_counts()
for label, value in name_counts_SibSp.items():
    entries = all_df_x_0[all_df_x_0['Fname'] == label]
    if (entries.shape[0] > 1 and (not (entries['Title'] == 'Mrs').any())) or \
       (entries.shape[0] == 1 and entries['Title'].values[0] == 'Mrs'):
            Pas_wSib.extend(entries['PassengerId'].values.tolist())
    else:
        Pas_wSib.extend(
            entries[(entries['Title'] == 'Miss')|(entries['GrSize'] == 1)]['PassengerId'].values.tolist())

# Search for Mrs-es with parents
Mrs_wPar = []
all_df_x_y = all_df[all_df['Parch'] > 0]
name_counts_Parch = all_df_x_y['Fname'].value_counts()
for label, value in name_counts_Parch.items():
    entries = all_df_x_y[all_df_x_y['Fname'] == label]
    if entries.shape[0] == 1:
        if entries['Title'].values[0] == 'Mrs' and entries['Age'].values[0] <= 30:
            Mrs_wPar.extend(entries['PassengerId'].values.tolist())

def get_features(row):

    features = pd.Series(0, index = ['isAlone','wSib','wSp','wCh','wPar'])

    if row['PassengerId'] in Pas_wSib:
        features['wSib'] = 1
    else:
        if row['FamSize'] == 1:
            features['isAlone'] = 1
        elif (row['SibSp'] != 0) & (row['Parch'] == 0):
            features['wSp'] = 1
        else:
            if  ( (row['Title']=='Mrs')&(not row['PassengerId'] in Mrs_wPar) )| \
                ( (row['Title']=='Mr')&(not row['PassengerId'] == 680)&
                                        ( ((row['Pclass']==1)&(row['Age']>=30))|
                                          ((row['Pclass']==2)&(row['Age']>=25))|
                                          ((row['Pclass']==3)&(row['Age']>=20)) ) ):
                features['wCh'] = 1
            else:
                features['wPar'] = 1

    return features

all_df[['isAlone','wSib','wSp','wCh','wPar']] = all_df.apply(lambda s: get_features(s), axis = 1)

# Add Per Person Fare
all_df.insert(15,'PerFare','')
all_df['Fare'].fillna(0.0, inplace=True)
all_df['PerFare'] = all_df.apply(lambda s: s['Fare']/s['GrSize'], axis=1)

# Fill Fare
perfare_medians = all_df.groupby(['Pclass'])['PerFare'].median()
all_df['PerFare'] = all_df.apply(lambda s: perfare_medians.loc[s['Pclass']]
    if s['PerFare'] == 0.0 else s['PerFare'], axis=1)
all_df['Fare'] = all_df.apply(lambda s: s['PerFare']*s['GrSize']
    if s['Fare'] == 0.0 else s['Fare'], axis = 1)

# Fill Age
age_medians = all_df.groupby(['Pclass','Title','isAlone','wSib','wSp','wCh','wPar'])['Age'].median()
all_df['Age'] = all_df.apply(lambda s:
    age_medians.loc[s['Pclass']].loc[s['Title']].loc[s['isAlone']].loc[s['wSib']]. \
                                 loc[s['wSp']].loc[s['wCh']].loc[s['wPar']]
    if pd.isnull(s['Age']) else s['Age'], axis=1)
all_df.loc[all_df['PassengerId'] == 1231,'Age'] = \
    age_medians.loc[3].loc['Master'].loc[0].loc[0].loc[0].loc[0].loc[1]

# Fill Embarked
all_df['Embarked'].fillna(all_df['Embarked'].value_counts().index[0], inplace=True)

# Select and convert categorial features into numerical ones
all_df['Sex'] = all_df['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
all_df[['Age','PerFare']] = all_df[['Age','PerFare']].astype(int)
all_df_dummies =  pd.get_dummies(all_df, columns = ['Title','Pclass','Embarked'], prefix=['Title','Pclass','Embarked'])

featr_drop = ['Fname','Name','Deck','Cabin','Ticket','Fare','SibSp','Parch','FamSize','GrSize']
all_df_dummies = all_df_dummies.drop(featr_drop, axis = 1)

# Form train and test sets
X_train = all_df_dummies.iloc[:891,:].drop(['PassengerId','Survived'], axis = 1)
y_train = all_df_dummies.iloc[:891,:]['Survived']
X_test = all_df_dummies.iloc[891:,:].drop(['PassengerId','Survived'], axis = 1)
X_test_Id = all_df_dummies.iloc[891:,:]['PassengerId']

# Perform scaling
scaler = StandardScaler()
scaler.fit(X_train[['Age','PerFare']])
X_train[['Age','PerFare']] = scaler.transform(X_train[['Age','PerFare']])
X_test[['Age','PerFare']] = scaler.transform(X_test[['Age','PerFare']])

# Cross-validation parameters
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1,)

# Support Vector Classifier score
alg_svm = SVC(C=1.0)
scores = cross_val_score(alg_svm, X_train, y_train, cv=cv)
print("Accuracy (SVM): {}/{}".format(scores.mean(), scores.std()))

# Fit, predict and generate submission
alg_svm.fit(X_train, y_train)
predictions = alg_svm.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": X_test_Id,
    "Survived": predictions.astype(int)
})

submission.to_csv("titanic-submission.csv", index=False)

# Test routine
# g = sns.FacetGrid(all_df[all_df['FamSize'] == 2], col='Parch')
# g.map(plt.hist, 'Age', bins=20)
# all_df[(all_df['SibSp'] == 0) & (all_df['Parch'] == 2)].to_csv('test_0_2.csv')
# all_df.groupby(['Pclass','Title','isAlone','wSibSp','wCh','wPar'])['Age'].agg(['count','median'])
# all_df.groupby(['Pclass','Title','isAlone','wSibSp','wCh','wPar'])['Survived'].describe()
# all_df.groupby(['Pclass','Title','isAlone','wSib','wSp','wCh','wPar'])['Survived'].agg(['count','mean'])
# all_df.groupby(['Pclass','Title','isAlone','wSib','wSp','wCh','wPar','Survived'])['Age'].agg(['count','mean'])

# xs = all_df[(all_df['Title'] != 'Mr') & (all_df['Pclass'] == 3) & (all_df['wPar'] == 1)]
# markers = ('x', 'o')
# colors = ('black', 'black')
# for idx in [0,1]:
#     plt.scatter(xs[xs['Survived'] == idx].Age, xs[xs['Survived'] == idx].PerFare, \
#                 alpha = 0.5, c = colors[idx], marker = markers[idx], label = idx)