# ML-Assignment-2-Titanic-_Survival-_Prediction

#Reading Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/traintitanic.csv')
df.head()

test_df = pd.read_csv('/content/drive/MyDrive/testtitanic.csv')
test_df.head()

#EDA & Some Cleaning

df.info()

df.describe()

#Duplicate Values

# Check duplicates
df.duplicated().sum()

#Missing Values

# Check Missing Values
df.isnull().sum()

round(df.isna().mean() * 100 ,2)

dtype: float64

df['Embarked'].describe()

df['Embarked'].value_counts()

df['Embarked'].mode()[0]

#df.dropna(subset=['Embarked'], inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

df.isnull().sum()

df.Age.hist(bins=20)

df[df['Sex'] == 'female'].Age.median()

df[df['Sex'] == 'male'].Age.median()

print(df[df['Pclass'] == 1].Age.median())
print(df[df['Pclass'] == 2].Age.median())
print(df[df['Pclass'] == 3].Age.median())

print(df[(df['Pclass'] == 1) & (df['Sex'] == 'female')].Age.median())
print(df[(df['Pclass'] == 1) & (df['Sex'] == 'male')].Age.median())
print(df[(df['Pclass'] == 2) & (df['Sex'] == 'female')].Age.median())
print(df[(df['Pclass'] == 2) & (df['Sex'] == 'male')].Age.median())
print(df[(df['Pclass'] == 3) & (df['Sex'] == 'female')].Age.median())
print(df[(df['Pclass'] == 3) & (df['Sex'] == 'male')].Age.median())

c1_f_age = 35
c1_m_age = 40
c2_f_age = 28
c2_m_age = 30
c3_f_age = 21
c3_m_age = 25

def impute_age(x):
    if pd.isnull(x['Age']):
        if x['Pclass'] == 1 and x['Sex'] == 'female':
             return c1_f_age
        elif x['Pclass'] == 1 and x['Sex'] == 'male':
             return c1_m_age
        elif x['Pclass'] == 2 and x['Sex'] == 'female':
             return c2_f_age
        elif x['Pclass'] == 2 and x['Sex'] == 'male':
             return c2_m_age
        elif x['Pclass'] == 3 and x['Sex'] == 'female':
             return c3_f_age
        else:
             return c3_m_age
    else:

        return x['Age']

df['Age'] = df.apply(lambda row : impute_age(row), axis=1)

df['Age']

# Check Missing Values
df.isnull().sum()

df.head()

df.isnull().sum()

#Drop Columns

# Drop Unnecessary columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

#Rename Columns

# Rename Columns
#df.rename(columns={'Survived': 'survived', 'Pclass': 'pclass', 'SibSp': 'brothers', 'Parch': 'parch'}, inplace=True)
df.columns = df.columns.str.lower()
#Create New Feature "family_size"

df['family_size'] = df['sibsp'] + df['parch']

df.sample()

df.describe()

df.describe(include='object')

#pclass and fare are the two major features that contributed to survival

#Univariate Analysis

df.survived.value_counts()

df.survived.value_counts() / len(df)   # shape[0]

df.survived.value_counts(normalize=True)

sns.countplot(x='survived', data=df);

df.pclass.value_counts(normalize=True)

** More than 55% of the passengers were from 3rd class**

df.sibsp.value_counts(normalize=True)

** More than 68% of the passengers had no siblings or spouse**

def explore_categorical(df, col):
    print(f'### {col} ###')
    print(df[col].value_counts(normalize=True))
    sns.countplot(x=col, data=df);
    plt.show()

for col in ['pclass', 'sibsp', 'parch', 'embarked', 'sex', 'family_size']:
    explore_categorical(df, col)

def explore_continuous(df, col):
    print(f'### {col} ###')
    print(df[col].describe())
    sns.histplot(x=col, data=df);
    plt.show()

for col in ['age', 'fare']:
    explore_continuous(df, col)

#Bivariate Analysis

df.sample()

df.groupby('pclass').survived.mean()

# plot class vs survived
sns.barplot(x='pclass', y='survived', data=df);

sns.barplot(x='pclass', y='survived', data=df, ci=None);
# plot horizontal line for mean
plt.axhline(df.survived.mean(), color='black', linestyle='--');
plt.show()

# Survival rate for each group
def survival_rate(df, col):
    print(df.groupby(col).survived.mean())
    sns.barplot(x=col, y='survived', data=df, ci=None);
    #plot horizontal line for overall survival rate
    plt.axhline(df.survived.mean(), color='black', linestyle='--')
    plt.show()

for col in ['pclass', 'sibsp', 'parch', 'embarked', 'sex', 'family_size']:
    survival_rate(df, col)

#How many passengers were alone? What is their Survival Rate ?

df[df['family_size'] ==0].shape[0]

df[df['family_size'] ==0].count()

df[df['family_size'] ==0]['family_size'].count()

df['family_size'].value_counts()

df['family_size'].value_counts()[0]

df['family_size'].value_counts().loc[0]

df['family_size'].value_counts(normalize= True)[0]

df[df['family_size'] ==0].survived.mean()
0.30353817504655495

#What are top 3 categories from "family_size" have highest survival Rate ?

df.groupby('family_size').survived.mean()

df.groupby('family_size').survived.mean().sort_values()

df.groupby('family_size').survived.mean().sort_values(ascending = False)

df.groupby('family_size').survived.mean().sort_values(ascending = False).head(3)

df.groupby('family_size').survived.mean().nlargest(3)

sns.histplot(x='age', data=df, hue='survived');

sns.histplot(x='fare', data=df, hue='survived');

sns.histplot(x='fare', data=df, hue='survived', multiple='stack');

df_survived = df[df.survived == 1]
df_died = df[df.survived == 0]

# Subplots for age distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(x='age', data=df_survived, ax=ax[0], kde=True, color='green')
sns.histplot(x='age', data=df_died, ax=ax[1], kde=True, color='red')
ax[0].set_title('Survived')
ax[1].set_title('Died')
plt.show()

# Subplots for fare distribution
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(x='fare', data=df_survived, ax=ax[0], kde=True, color='green')
sns.histplot(x='fare', data=df_died, ax=ax[1], kde=True, color='red')
ax[0].set_title('Survived')
ax[1].set_title('Died')
plt.show()

df.describe()[['age', 'fare']]

sns.boxplot( x='age', data=df);

sns.boxplot( x='fare', data=df);

#Many outliers in the fare

#We can use the IQR method to remove them.

#We can also choose a limit according to the distribution of the data.

#Outliers Detection

# Remove outliers
def remove_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return df[(df[col] > lower_bound) & (df[col] < upper_bound)]

df = df[df.fare < 300]
# df = remove_outliers(df, 'fare')

sns.boxplot( x='fare', data=df);

df.fare.describe()

# Split Age into groups
df['age_group'] = pd.cut(df.age, bins=[0, 18, 24, 37, 82], labels=['child', 'young', 'adult', 'senior'])

# Split Fare into groups
df['fare_group'] = pd.cut(df.fare, bins=[-0.99, 8, 15, 35, 265], labels=['low', 'medium', 'high', 'very high'])

for col in ['age_group', 'fare_group']:
    survival_rate(df, col)

#Multivariate Analysis

sns.pairplot(df, hue='survived');

# Import pandas package
import pandas as pd
 
df1 = pd.DataFrame(df)
 
# Remove two columns name is 'C' and 'D'
df2=df1.drop(['sex', 'embarked','age_group','fare_group'], axis=1)

df2.corr()['survived']

plt.figure(figsize=(10, 8))
sns.heatmap(df2.corr(), annot=True);

# Check the correlation between the numerical features
# Heatmap
sns.heatmap(df2.corr(), annot=True, cmap='RdYlGn', center=0)

# Reascending
asc_num_features = df2.corr()['survived'].apply(abs).sort_values(ascending=False).drop('survived')
sns.barplot(x=asc_num_features, y=asc_num_features.index)
plt.title('Absolute Correlation with the target')
plt.show()

# Correlation with the target
df2.corr()['survived'].sort_values(ascending=False)

we noticed High correlations between (Passenger Class and Fare Class)

Medium correlations between (passenger class and age), (survived and Fare ), (passenger class and survived) and (parch and sibsp)

sns.barplot(x='pclass', y='survived', hue= 'sex', data=df, ci=None);

sns.barplot(x='embarked', y='survived', hue= 'sex', data=df, ci=None);

#Conclusion

# 8 Subplots
fig, ax = plt.subplots(2, 4, figsize=(15, 8))
for i, col in enumerate(['pclass', 'sibsp', 'parch', 'family_size', 'age_group', 'fare_group', 'embarked', 'sex']):
    sns.barplot(x=col, y= 'survived', data=df, ci=None, ax=ax[i//4, i%4])
    ax[i//4, i%4].axhline(df.survived.mean(), color='black', linestyle='--')

plt.tight_layout()
plt.show()

female_df = df[df.sex == 'female']
male_df = df[df.sex == 'male']

female_df.survived.value_counts(normalize=True)

female_df.groupby('pclass').survived.mean()

sns.barplot(x='pclass', y='survived', data=female_df, ci=None);
plt.axhline(female_df.survived.mean(), color='black', linestyle='--')
plt.show()

male_df.survived.value_counts(normalize=True)

male_df.groupby('pclass').survived.mean()

sns.barplot(x='pclass', y='survived', data=male_df, ci=None);
plt.axhline(male_df.survived.mean(), color='black', linestyle='--')

sex_class = pd.merge(female_df.groupby('pclass').survived.mean(), male_df.groupby('pclass').survived.mean(), on='pclass')
sex_class

sex_class.rename(columns= {'survived_x': 'female_survived', 'survived_y': 'male_survived'}, inplace=True)

df.groupby(['pclass', 'sex']).survived.mean()

df.groupby(['age_group', 'sex']).survived.mean()

pd.DataFrame(df.groupby(['age_group', 'sex']).survived.mean())

age_sex = pd.DataFrame(df.groupby(['age_group', 'sex']).survived.mean()).sort_values(by='survived')
age_sex

age_sex = age_sex.reset_index()
age_sex

age_sex[age_sex['sex']== 'female']

age_sex[age_sex['sex']== 'female'].iloc[0]['age_group']

age_sex[age_sex['sex']== 'female'].iloc[-1]

age_sex[age_sex['sex']== 'female'].iloc[-1]['age_group']

age_sex[age_sex['sex']== 'male'].iloc[0]['age_group']

age_sex[age_sex['sex']== 'male'].iloc[-1]['age_group']

#Insights

#The higher the class, the higher the survival rate

#The higher the fare, the higher the survival rate

#Females had a higher survival rate

#Data Preprocessing

df.head()

test_df
]
#Columns to DROP:SibSp,Parch,Fare,Age,Fare_group

#Columns to Create: Age_class,Fare_class,married_women.

#Pclass,Embarked(i suggest to drop from beginning),Title: OneHotEncoding

#Sex,Family_size: Label Encoder.

#Fare_class,Age_class ready

6.1 Quick Pipeline

#Create the Model

from sklearn.ensemble import RandomForestClassifier

X = df.drop(['survived','sex','age_group','fare_group','embarked'] , axis = 1)
y = df['survived']

from sklearn.model_selection import train_test_split
#splitting the data into train and validation sets
X_train , X_val , y_train , y_val = train_test_split( X, y , test_size=0.2 , random_state = 42)

print(X_train.shape)
print(X_val.shape)

from sklearn.ensemble import RandomForestClassifier
RandomForestClassifier()

fitted_rf = RandomForestClassifier(
    n_estimators=1750,max_depth=7,min_samples_split=6,
    min_samples_leaf=6,max_features='auto',oob_score=True,random_state=77,n_jobs=-1)

fitted_rf.fit(X_train , y_train)

from sklearn.feature_selection import SelectFromModel
select = SelectFromModel(estimator=fitted_rf,prefit=True)
selected_data = select.fit_transform(X_train,y_train)

from sklearn.model_selection import cross_val_score
cv = cross_val_score(estimator=fitted_rf,X=selected_data,y=y_train,
                     cv=5,n_jobs= -1)
print(cv)
print('Mean Accuarcy: ',cv.mean())


feature_importances = fitted_rf.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns,
                                      'Importance': feature_importances}).sort_values(by='Importance',
                                                                                      ascending=False)
plt.figure(figsize=(9, 5))
a = sns.barplot(y=feature_importance_df['Feature'][:18], x=feature_importance_df['Importance'][:18]*100,palette='Reds_r')
heights = [str(round(height ,1))+'%' for height in a.containers[0].datavalues]
a.bar_label(a.containers[0], labels=heights, label_type='center')
plt.title('Model Feature Importance')
plt.show()

df.sample(1)

X_train.sample(1)

6.2 Full Pipeline

7- BEAST Model 💡

from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifierCV
model = RidgeClassifierCV()
model.fit(X_train,y_train)

tr_preds = model.predict(X_train)
print('Train Accuracy',accuracy_score(y_train,tr_preds))
ts_preds = model.predict(X_val)
print('Test Accuracy',accuracy_score(y_val,ts_preds))

test_data = pd.read_csv('/content/drive/MyDrive/testtitanic.csv')
passengersID = test_data['PassengerId']

X_train

test_df

c1_f_age = 35
c1_m_age = 40
c2_f_age = 28
c2_m_age = 30
c3_f_age = 21
c3_m_age = 25

def impute_age(x):
    if pd.isnull(x['Age']):
        if x['Pclass'] == 1 and x['Sex'] == 'female':
             return c1_f_age
        elif x['Pclass'] == 1 and x['Sex'] == 'male':
             return c1_m_age
        elif x['Pclass'] == 2 and x['Sex'] == 'female':
             return c2_f_age
        elif x['Pclass'] == 2 and x['Sex'] == 'male':
             return c2_m_age
        elif x['Pclass'] == 3 and x['Sex'] == 'female':
             return c3_f_age
        else:
             return c3_m_age
    else:

        return x['Age']

test_df['Age'] = test_df.apply(lambda row : impute_age(row), axis=1)

test_df.isnull().sum()

#Dropping the columns which are not needed
test_df = test_df.drop(['PassengerId' , 'Name' , 'Sex','Ticket','Cabin','Embarked'] , axis  = 1)
test_df.head()

# Rename Columns
#df.rename(columns={'Survived': 'survived', 'Pclass': 'pclass', 'SibSp': 'brothers', 'Parch': 'parch'}, inplace=True)
test_df.columns = test_df.columns.str.lower()

test_df['family_size'] = test_df['sibsp'] + test_df['parch']

test_df['fare']

test_df.isnull().sum()

import pandas as pd

testdf = test_df.dropna()

testdf.isnull().sum()

predictions = model.predict(testdf)

predictions

model.get_params()

8-Error analysis

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
ypreds = cross_val_predict(RidgeClassifierCV(),X_train,y_train,cv=3)
cf = confusion_matrix(y_train,ypreds)
plt.figure(figsize=(6,4))
sns.heatmap(cf,annot=True,fmt='d' ,cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted');plt.ylabel('True');plt.title('Ridge Classifier Error Analysis')
plt.show()

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
print('Accuracy Score: ',accuracy_score(y_train,ypreds),end='\n\n')
print('Precision Score: ',precision_score(y_train,ypreds))
print('Recall Score: ',recall_score(y_train,ypreds))
print('F1 Score: ',f1_score(y_train,ypreds))
print('ROC AUC Score: ',roc_auc_score(y_train,ypreds))

log_model = LogisticRegression()
log_model.fit(X_train,y_train)

y_scores = cross_val_predict(RidgeClassifierCV(),X_train,y_train,method='decision_function',cv=3)

from sklearn.metrics import precision_recall_curve
precisions,recalls,thresholds = precision_recall_curve(y_train,y_scores)

plt.figure(figsize=(9,5))
plt.plot(thresholds,precisions[:-1],'b--',label='Precision',linewidth=2)
plt.plot(thresholds,recalls[:-1],'g-',label='Recalls',linewidth=2)
plt.vlines(.5,0,1.0,'k','dotted',label='threshold at 0.5')
plt.title('Precision-Recall');plt.legend();plt.grid();plt.show()

plt.figure(figsize=(8,5))
plt.plot(recalls,precisions,linewidth=2,label='Precision/Recall Curve')
plt.legend();plt.grid();plt.ylabel('Precision');plt.xlabel('Recall');
plt.show()

idx_for_90_precision = (precisions>=.90).argmax()
threshold_for_90_pr = thresholds[idx_for_90_precision]
y_prec_90 = (y_scores>=threshold_for_90_pr)
print('Precision Score: ',precision_score(y_train,y_prec_90))
print('Recall Score: ',recall_score(y_train,y_prec_90))

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train,y_scores)

idx_for_threshold_90 = (thresholds<=threshold_for_90_pr).argmax()
tpr_90,fpr_90 = tpr[idx_for_threshold_90],fpr[idx_for_threshold_90]

from sklearn.model_selection import cross_val_predict
y_scoresi = cross_val_predict(RandomForestClassifier(),X_train,y_train,method='predict_proba',cv=3)
y_scoresi

from sklearn .metrics import roc_auc_score
roc_auc_score(y_train,y_scores)

9-voting classifier

from sklearn.ensemble import VotingClassifier
m1 = [LogisticRegression(random_state=33),CalibratedClassifierCV()]
votingcl = VotingClassifier(
    estimators=[(f'{k}',k)for k in m1],
    voting='soft',n_jobs=-1,
    verbose=1,
    weights=[5,2],
)
votingcl.fit(X_train,y_train)

tr_preds = votingcl.predict(X_train)
print('Train Accuracy',accuracy_score(y_train,tr_preds))
ts_preds = votingcl.predict(X_val)
print('Test Accuracy',accuracy_score(y_val,ts_preds))

ridge = RidgeClassifierCV(cv=5)
ridge.fit(X_train,y_train)
probstrain1 = ridge.decision_function(X_train)
probstest1 = ridge.decision_function(X_val)

from sklearn.svm import LinearSVC
linSVC = LinearSVC(random_state=3)
linSVC.fit(X_train,y_train)
probstrain2 = linSVC.decision_function(X_train)
probstest2 = linSVC.decision_function(X_val)

probstrain = (1.2*probstrain1+.8*probstrain2)/2
probstest = (1.2*probstest1+.8*probstest2)/2

tr_preds = (probstrain>0)
print('Train Accuracy',accuracy_score(y_train,tr_preds))
ts_preds = (probstest>0)
print('Test Accuracy',accuracy_score(y_val,ts_preds))

predictions1 = ridge.decision_function(testdf)
predictions2 = linSVC.decision_function(testdf)
predictions = ((1.2*predictions1+.8*predictions2)/2>0).astype(int)

predictions

predictions = votingcl.predict(testdf)

predictions

tr_preds = ((probstrain>0).astype(int)+votingcl.predict(X_train)>0).astype(int)
ts_preds = ((probstest>0).astype(int)+votingcl.predict(X_val)>0).astype(int)

print('Train Accuracy',accuracy_score(y_train,tr_preds))
print('Test Accuracy',accuracy_score(y_val,ts_preds))

predictions1 = ridge.decision_function(testdf)
predictions2 = linSVC.decision_function(testdf)
predictions3 = ((1.3*predictions1+.7*predictions2)/2>0).astype(int)

predictions= (predictions3+votingcl.predict(testdf)>1).astype(int)

predictions
