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
             1254: pd.Series([1,0], index=['SibSp', 'Parch'],),
             1274: pd.Series([1,0], index=['SibSp', 'Parch'],),
             539: pd.Series([1,0], index=['SibSp', 'Parch'],)
             }

all_df[['SibSp','Parch']] = all_df.apply(lambda s: corr_dict[s['PassengerId']]
    if s['PassengerId'] in [248,313,418,756,1041,1130,1170,1254,1274,539] else s[['SibSp','Parch']], axis = 1)

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
        Pas_wSib.extend( \
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

# Make Group Size bins
all_df['GrSize'] = pd.cut(all_df['GrSize'], bins = [0,1,4,11], labels = False)

# Make FamSize bins
all_df['FamSize'] = pd.cut(all_df['FamSize'], bins = [0,4,11], labels = False)

# Fill Age
age_medians = all_df.groupby(['Pclass','Title','isAlone','wSib','wSp','wCh','wPar','GrSize'])['Age'].median()
all_df['Age'] = all_df.apply(lambda s:
    age_medians.loc[s['Pclass']].loc[s['Title']].loc[s['isAlone']].loc[s['wSib']]. \
                                 loc[s['wSp']].loc[s['wCh']].loc[s['wPar']].loc[s['GrSize']]
    if pd.isnull(s['Age']) else s['Age'], axis=1)
all_df.loc[all_df['PassengerId'] == 1231,'Age'] = \
    age_medians.loc[3].loc['Master'].loc[0].loc[0].loc[0].loc[0].loc[1].loc[1]

# Fill Embarked
all_df['Embarked'].fillna(all_df['Embarked'].value_counts().index[0], inplace=True)

# Merge isAlone and wSib
all_df['isAlwSib'] = all_df.apply(lambda s: 1 if (s['isAlone'] == 1)|(s['wSib'] == 1) else 0 ,axis = 1)

# Form train and test sets
df_train = all_df.iloc[:891,:]
df_test = all_df.iloc[891:,:]

# Select and convert categorial features into numerical ones
featr = ['PassengerId','Survived','Age','PerFare',\
         'Ch12','Ch3','Fem12','Fem3wCh','Fem3r','Male1wSp','Male1r','Male2','Male3isAlwSib','Male3r']

def get_category(row):
    category = pd.Series(0, index = featr)
    category['PassengerId'] = row['PassengerId']
    if not pd.isnull(row['Survived']):
        category['Survived'] = row['Survived']
    if row['Pclass'] == 1:
         category['PerFare'] = int(row['PerFare'])
         category['Age'] = 0
    elif row['Pclass'] == 3:
         category['PerFare'] = 0
         category['Age'] = int(row['Age'])

    if (row['Title'] == 'Master' or (row['Title'] == 'Miss' and row['wPar'] == 1)):
        if row['Pclass'] in [1,2]:
            category['Ch12'] = 1
        else:
            category['Ch3'] = 1
    else:
        if row['Sex'] == 'female':
            if row['Pclass'] in [1,2]:
                category['Fem12'] = 1
            else:
                if row['wCh'] == 1:
                    category['Fem3wCh'] = 1
                else:
                    category['Fem3r'] = 1
        else:
            if row['Pclass'] == 1:
                if row['wSp'] == 1:
                    category['Male1wSp'] = 1
                else:
                    category['Male1r'] = 1
            elif row['Pclass'] == 2:
                category['Male2'] = 1
            else:
                if row['isAlone'] == 1 or row['wSib'] == 1:
                    category['Male3isAlwSib'] = 1
                else:
                    category['Male3r'] = 1

    return category

cat_df_train = all_df.iloc[:891,:].apply(lambda s: get_category(s) ,axis = 1)
cat_df_test = all_df.iloc[891:,:].apply(lambda s: get_category(s) ,axis = 1)

# Baseline Model
def get_survived(row):
    if (row['Fem12'] == 1)|(row['Ch12'] == 1)|(row['Fem3r'] == 1)|(row['Fem3wCh'] == 1):
        survived = 1
    else:
        survived = 0

    return survived

def get_survived_s(row):
    if row['Pclass'] == 1:
        if row['Title'] == 'Mr':
            if row['Deck'] == 'E':
                survived = 1
            else:
                survived = 0
        else:
            survived = 1
    elif row['Pclass'] == 2:
        if row['Title'] == 'Mr':
            survived = 0
        else:
            survived = 1
    else:
        if row['Title'] == 'Mr' or row['FamSize'] > 0 or \
                (row['Title'] == 'Miss' and row['Embarked'] == 'S'):
            survived = 0
        else:
            survived = 1

    return survived

pred_df_train = pd.DataFrame( {'PassengerId': train_df['PassengerId'], 'Survived': 0} )
pred_df_test = pd.DataFrame( {'PassengerId': test_df['PassengerId'], 'Survived': 0} )

pred_df_train['Survived'] = df_train.apply(lambda s: get_survived_s(s), axis = 1)
pred_df_test['Survived'] = df_test.apply(lambda s: get_survived_s(s), axis = 1)

score = metrics.accuracy_score(pred_df_train['Survived'], train_df['Survived'])
print('Accuracy: {}'.format(score))
# Submission
pred_df_test.to_csv('submission.csv', index=False)

# Cross-validation parameters
# cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

# Random Forest classifier
# alg_tree = DecisionTreeClassifier()

# Submission
# alg_tree.fit(cat_df_train, train_df['Survived'])
# print(alg_tree.score(cat_df_train, train_df['Survived']))

# predictions = alg_tree.predict(cat_df_test)

# Test routine
# g = sns.FacetGrid(all_df[all_df['FamSize'] == 2], col='Parch')
# g.map(plt.hist, 'Age', bins=20)
# all_df[(all_df['SibSp'] == 0) & (all_df['Parch'] == 2)].to_csv('test_0_2.csv')
# all_df.groupby(['Pclass','Title','isAlone','wSibSp','wCh','wPar'])['Age'].agg(['count','median'])
# all_df.groupby(['Pclass','Title','isAlone','wSibSp','wCh','wPar'])['Survived'].describe()
# all_df[all_df['Pclass'] == 3].groupby(['Title'])['Survived'].agg(['count','size','mean'])
# all_df.groupby(['Pclass','Title','isAlone','wSib','wSp','wCh','wPar','Survived'])['Age'].agg(['count','mean'])
# all_df[all_df['Pclass'] == 1].groupby(['Title','wSp','FamSize','Deck'])['Survived'].agg(['count','size','mean'])
# print(all_df[all_df['Pclass'] == 1].groupby(['Title','FamSize','Deck'])['Survived'].agg(['count','size','mean']))

# xs = all_df[(all_df['Title'] == 'Miss') & (all_df['Pclass'] == 3) & \
#             (all_df['FamSize'] < 2) & (all_df['wPar'] == 1) ]
# markers = ('x', 'o')
# colors = ('black', 'black')
# for idx in [0,1]:
#     plt.scatter(xs[xs['Survived'] == idx].Age, xs[xs['Survived'] == idx].PerFare, \
#                 alpha = 0.5, c = colors[idx], marker = markers[idx], label = idx)