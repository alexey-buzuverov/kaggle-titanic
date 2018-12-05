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
from fancyimpute import KNN
from fancyimpute import IterativeImputer
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings('ignore')

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
             1274: pd.Series([1,0], index=['SibSp', 'Parch']),
             539: pd.Series([1,0], index=['SibSp', 'Parch']),
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
all_df['Deck'] = all_df.apply(lambda s: 0 if s['Deck'] == 'U' else 1,axis = 1)

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

# Select and convert categorial features into numerical ones (1)
all_df['Sex'] = all_df['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
all_df_dummies =  pd.get_dummies(all_df, columns = ['Title','Pclass','Embarked'],\
                                 prefix=['Title','Pclass','Embarked'], drop_first = True)
featr_drop = ['Fname','Name','Cabin','Ticket','Fare','PassengerId','Survived','SibSp','Parch']
all_df_dummies = all_df_dummies.drop(featr_drop, axis = 1)

# KNN imputation
all_df_dummies_i = pd.DataFrame(data=KNN(k=3, verbose = False).fit_transform(all_df_dummies).astype(int),
                            columns=all_df_dummies.columns, index=all_df_dummies.index)

# Convert categorial features into numerical ones (2)
all_df_dummies_i['FamSize'] = pd.cut(all_df_dummies_i['FamSize'], bins = [0,4,11], labels = False)
all_df_dummies_i['isAlwSib'] = \
    all_df_dummies_i.apply(lambda s: 1 if (s['isAlone'] == 1)|(s['wSib'] == 1) else 0 ,axis = 1)
# all_df_dummies_i['wSpwCh'] = \
#     all_df_dummies_i.apply(lambda s: 1 if (s['wSp'] == 1)|(s['wCh'] == 1) else 0 ,axis = 1)
# all_df_dummies_i =  pd.get_dummies(all_df_dummies_i, columns = ['wSp','wCh','wPar','isAlwSib','FamSize','Deck'],\
#                                 prefix=['wSp','wCh','wPar','isAlwSib','FamSize','Deck'])
all_df_dummies_i = all_df_dummies_i.drop(['Sex','isAlone','wSib','GrSize'], axis = 1)

# Form train and test sets
X_train = all_df_dummies_i.iloc[:891,:]
y_train = all_df.iloc[:891,:]['Survived']
X_test = all_df_dummies_i.iloc[891:,:]
X_test_Id = all_df.iloc[891:,:]['PassengerId']
X_train.to_csv('train.csv')

# Perform scaling
scaler = StandardScaler()
scaler.fit(X_train[['Age']])
X_train['Age'] = scaler.transform(X_train[['Age']])
X_test['Age'] = scaler.transform(X_test[['Age']])
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
lr_grid = {'C': list(np.linspace(0.1,2,20))}
lr_search = GridSearchCV(estimator = LogisticRegression(), param_grid = lr_grid,
               cv = cv, refit=True, n_jobs=1)
# lr_search.fit(X_train, y_train)
# lr_best = lr_search.best_estimator_
# print("Accuracy CV: {}, std: {}, with params {}"
#       .format(lr_search.best_score_, lr_search.cv_results_['std_test_score'][lr_search.best_index_],
#               lr_search.best_params_))

# Support Vector Classifier
svm_grid = {'C': [10], 'gamma': ['auto']}
# svm_grid = {'C': [12], 'gamma': ['auto']}
svm_search = GridSearchCV(estimator = SVC(), param_grid = svm_grid,
               cv = cv, refit=True, n_jobs=1)
svm_search.fit(X_train, y_train)
svm_best = svm_search.best_estimator_
print("Accuracy CV: {}, std: {}, with params {}"
       .format(svm_search.best_score_, svm_search.cv_results_['std_test_score'][svm_search.best_index_],
               svm_search.best_params_))

# K-nearest Neighbors
knn_grid = {'algorithm': ['auto'], 'weights': ['uniform', 'distance'], 'leaf_size': list(range(1,50,5)),
               'metric': ['minkowski'], 'n_neighbors': list(range(2,6))}
knn_search = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = knn_grid,
                cv=cv, refit=True, n_jobs=1)
# knn_search.fit(X_train, y_train)
# knn_best = knn_search.best_estimator_
# print("Accuracy CV: {}, std: {}, with params {}"
#        .format(knn_search.best_score_, knn_search.cv_results_['std_test_score'][knn_search.best_index_],
#                knn_search.best_params_))

# Decision Tree
dt_grid = {'max_depth': list(range(2,10)), 'min_samples_split': list(range(2,10)),
           'min_samples_leaf': list(range(2,10))}
dt_search = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid = dt_grid, scoring = 'roc_auc',
               cv = cv, refit=True, n_jobs=1)
# dt_search.fit(X_train, y_train)
# dt_best = dt_search.best_estimator_
# print("Accuracy CV: {}, std: {}, with params {}"
#       .format(dt_search.best_score_, dt_search.cv_results_['std_test_score'][dt_search.best_index_],
#               dt_search.best_params_))
# export_graphviz(dt_best, out_file = 'tree.dot', feature_names = predictors)

# Random Forest
rf_grid = {'n_estimators': [1000],
           'max_depth': [8],
           'min_samples_split': [10],
           'min_samples_leaf': [6]}
# rf_grid = {'n_estimators': [1000],
#             'max_depth': [12],
#             'min_samples_split': [8,10,12],
#             'min_samples_leaf': [4,6]}
rf_search = GridSearchCV(estimator = RandomForestClassifier(), param_grid = rf_grid,
               cv = cv, refit=True, n_jobs=1)
# rf_search.fit(X_train, y_train)
# rf_best = rf_search.best_estimator_
# print("Accuracy CV: {}, std: {}, with params {}"
#        .format(rf_search.best_score_, rf_search.cv_results_['std_test_score'][rf_search.best_index_],
#                rf_search.best_params_))

# Random Forest feature importance
# rf_importances = rf_best.feature_importances_
# plt.bar(range(len(rf_importances)), rf_importances)
# plt.xticks(range(len(rf_importances)), predictors, rotation='vertical')
# plt.show()

# Tree-based Gradient Boosting
xgb_grid = {'n_estimators': [30],
            'learning_rate': [0.1],
            'max_depth': [3],
            'min_child_weight': [5],
            'gamma': [0],
            'subsample': [0.9],
            'colsample_bytree': [0.9],
            'reg_alpha':[0.01]}
xgb_search = GridSearchCV(estimator = xgb.XGBClassifier(), param_grid = xgb_grid, scoring = 'roc_auc',
               cv = cv, refit=True, n_jobs=1)
# xgb_search.fit(X_train, y_train)
# xgb_best = xgb_search.best_estimator_
# print("Accuracy CV: {}, std: {}, with params {}"
#        .format(xgb_search.best_score_, xgb_search.cv_results_['std_test_score'][xgb_search.best_index_],
#                xgb_search.best_params_))

# Fit, predict and generate submission
alg_best = svm_best
predictions_train = alg_best.predict(X_train)
print('Accuracy Train: {}'
        .format(metrics.accuracy_score(y_train, predictions_train)))

predictions_test = alg_best.predict(X_test)
submission = pd.DataFrame({
    'PassengerId': X_test_Id,
    'Survived': predictions_test.astype(int)
})
submission.to_csv('submission.csv', index=False)

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