# Titanic Kaggle

import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, KFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
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
corr_dict = {248: pd.Series([0,1], index=['SibSp', 'Parch']),
             313: pd.Series([1,0], index=['SibSp', 'Parch']),
             418: pd.Series([0,0], index=['SibSp', 'Parch']),
             756: pd.Series([0,1], index=['SibSp', 'Parch']),
             1041: pd.Series([1,0], index=['SibSp', 'Parch']),
             1130: pd.Series([0,0], index=['SibSp', 'Parch']),
             1170: pd.Series([2,0], index=['SibSp', 'Parch']),
             1254: pd.Series([1,0], index=['SibSp', 'Parch']),
             1274: pd.Series([1, 0], index=['SibSp', 'Parch']),
             539: pd.Series([1, 0], index=['SibSp', 'Parch'])
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

# Make Group Size bins
all_df['GrSize'] = pd.cut(all_df['GrSize'], bins = [0,1,4,11], labels = False)

# Make Family Size bins
all_df['FamSize'] = pd.cut(all_df['FamSize'], bins = [0,4,11], labels = False)

# Fill Age
age_medians = all_df.groupby(['Pclass','Title','isAlone','wSib','wSp','wCh','wPar','GrSize'])['Age'].median()
all_df['Age'] = all_df.apply(lambda s:
    age_medians.loc[s['Pclass']].loc[s['Title']].loc[s['isAlone']].loc[s['wSib']]. \
                                 loc[s['wSp']].loc[s['wCh']].loc[s['wPar']].loc[s['GrSize']]
    if pd.isnull(s['Age']) else s['Age'], axis=1)
all_df.loc[all_df['PassengerId'] == 1231,'Age'] = \
    age_medians.loc[3].loc['Master'].loc[0].loc[0].loc[0].loc[0].loc[1].loc[1]

# Select and convert categorial features into numerical ones
all_df['Sex'] = all_df['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
all_df['isAlwSib'] = all_df.apply(lambda s: 1 if (s['isAlone'] == 1)|(s['wSib'] == 1) else 0 ,axis = 1)
# all_df[['Age','PerFare']] = all_df[['Age','PerFare']].astype(int)
all_df_dummies =  pd.get_dummies(all_df, columns = ['Title','Pclass','FamSize','Embarked'],\
                                 prefix=['Title','Pclass','FamSize','Embarked'])

featr_drop = ['Sex','Fname','Name','Deck','Cabin','Ticket','Fare','PerFare',
              'SibSp','Parch','GrSize','isAlone','wSib']
all_df_dummies = all_df_dummies.drop(featr_drop, axis = 1)

# Form train and test sets
X_train = all_df_dummies.iloc[:891,:].drop(['PassengerId','Survived'], axis = 1)
y_train = all_df_dummies.iloc[:891,:]['Survived']
X_test = all_df_dummies.iloc[891:,:].drop(['PassengerId','Survived'], axis = 1)
X_test_Id = all_df_dummies.iloc[891:,:]['PassengerId']
# X_train.to_csv('train.csv')

# Perform scaling
scaler = StandardScaler()
scaler.fit(X_train[['Age']])
X_train[['Age']] = scaler.transform(X_train[['Age']])
X_test[['Age']] = scaler.transform(X_test[['Age']])
predictors = list(X_train.columns.values)

# Feature importance
# selector = SelectKBest(f_classif, k=10)
# selector.fit(X_train, y_train)
# scores = -np.log10(selector.pvalues_)
# plt.bar(range(len(predictors)), scores)
# plt.xticks(range(len(predictors)), predictors, rotation='vertical')
# plt.show()

# Cross-validation parameters
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)

# Logistic Regression
lr_grid = {'C': list(np.linspace(0.1,1,10))}
lr_search = GridSearchCV(estimator = LogisticRegression(), param_grid = lr_grid, scoring = 'roc_auc',
               cv = cv, refit=True, n_jobs=1)
# lr_search.fit(X_train, y_train)
# lr_best = lr_search.best_estimator_
# print("Accuracy: {}, std: {}, with params {}"
#       .format(lr_search.best_score_, lr_search.cv_results_['std_test_score'][lr_search.best_index_],
#               lr_search.best_params_))

# Support Vector Classifier
svm_grid = {'C': list(range(1,10))}
svm_search = GridSearchCV(estimator = SVC(), param_grid = svm_grid, scoring = 'roc_auc',
               cv = cv, refit=True, n_jobs=1)
# svm_search.fit(X_train, y_train)
# svm_best = svm_search.best_estimator_
# print("Accuracy: {}, std: {}, with params {}"
#        .format(svm_search.best_score_, svm_search.cv_results_['std_test_score'][svm_search.best_index_],
#                svm_search.best_params_))

# K-nearest Neighbors
knn_grid = {'algorithm': ['auto'], 'weights': ['uniform', 'distance'], 'leaf_size': list(range(1,50,5)),
               'metric': ['minkowski'], 'n_neighbors': list(range(2,6))}
knn_search = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = knn_grid, scoring = 'roc_auc',
                cv=cv, refit=True, n_jobs=1)
# knn_search.fit(X_train, y_train)
# knn_best = knn_search.best_estimator_
# print("Accuracy: {}, std: {}, with params {}"
#        .format(knn_search.best_score_, knn_search.cv_results_['std_test_score'][knn_search.best_index_],
#                knn_search.best_params_))

# Decision Tree
dt_grid = {'max_depth': list(range(2,10)), 'min_samples_split': list(range(2,10)),
           'min_samples_leaf': list(range(2,10))}
dt_search = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid = dt_grid, scoring = 'roc_auc',
               cv = cv, refit=True, n_jobs=1)
# dt_search.fit(X_train, y_train)
# dt_best = dt_search.best_estimator_
# print("Accuracy: {}, std: {}, with params {}"
#       .format(dt_search.best_score_, dt_search.cv_results_['std_test_score'][dt_search.best_index_],
#               dt_search.best_params_))
# export_graphviz(dt_best, out_file = 'tree.dot', feature_names = predictors)

# Random Forest
# rf_grid = {'n_estimators': [50,100,200,250],
#            'max_depth': [4,8,12],
#            'min_samples_split': [4,8,12],
#            'min_samples_leaf': [4,8,12]}
rf_grid = {'n_estimators': [200],
            'max_depth': [8],
            'min_samples_split': [8],
            'min_samples_leaf': [8]}
rf_search = GridSearchCV(estimator = RandomForestClassifier(), param_grid = rf_grid, scoring = 'roc_auc',
               cv = cv, refit=True, n_jobs=1)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_
print("Accuracy: {}, std: {}, with params {}"
       .format(rf_search.best_score_, rf_search.cv_results_['std_test_score'][rf_search.best_index_],
               rf_search.best_params_))

# Tree-based Gradient Boosting
xgb_grid = {'n_estimators': [150, 200, 250],
            'max_depth': [1, 2, 3, 4],
            'learning_rate': [0.02, 0.05, 0.1]}
xgb_search = GridSearchCV(estimator = xgb.XGBClassifier(), param_grid = xgb_grid, scoring = 'roc_auc',
               cv = cv, refit=True, n_jobs=1)
# xgb_search.fit(X_train, y_train)
# xgb_best = xgb_search.best_estimator_
# print("Accuracy: {}, std: {}, with params {}"
#        .format(xgb_search.best_score_, xgb_search.cv_results_['std_test_score'][xgb_search.best_index_],
#                xgb_search.best_params_))

# Fit, predict and generate submission
predictions = rf_best.predict(X_test)
submission = pd.DataFrame({
    "PassengerId": X_test_Id,
    "Survived": predictions.astype(int)
})
submission.to_csv("submission.csv", index=False)

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