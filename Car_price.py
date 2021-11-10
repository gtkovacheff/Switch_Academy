import os
from numpy.lib.npyio import save
import pandas as pd
import numpy as np
from scipy.sparse.construct import random
import seaborn as sns
from matplotlib import pyplot as plt
import datatable as dt
from sklearn.model_selection import train_test_split


#import the data with data table and convert to pandas (it is faster)
df = dt.fread('Data/data.csv').to_pandas()

print(df.shape) #11914 x 16
print(df.head()) #head of the data 
print(df.tail()) #tail of the data

#metadata and stats
print(df.info())
print(df.describe())
 
#print column names
[print(x) for x in df]

#fix columns' name to be lower case and replace '\t' with '_'
df.columns = df.columns.str.lower().str.replace(' ', '_')

#preserve the string columns and the int columns
string_columns = df.select_dtypes(include=['object']).columns
int_columns = df.columns.difference(string_columns)


df[int_columns]
df[string_columns]


#EXPLORATORY DATA ANALYSIS
#show na values 
df.isnull().sum()

pd.DataFrame(df.isnull().sum()).reset_index().rename(columns={'index':'column',0:'sum_na'}).query('sum_na>0')

#Plot every numeric column
len(int_columns)
f, axes = plt.subplots(2,4)

#one way if we already know the structure of the subplots 
for i,x in enumerate(int_columns):
    # plt.figure(figsize=(16,9))
    # print(i)
    if i<=3: 
        k=0
    else: 
        k=1
        i = i-4
    sns.histplot(x=df[x], bins=20, color='blue', alpha=1, ax=axes[k, i])
    axes[k, i].set_xlabel(f'{x}')
    axes[k, i].set_ylabel('Frequency')
    axes[k, i].set_title(f'Histogram of the {x}')

f.suptitle('Distribution of all numeric colums', fontsize=30)
f.tight_layout()
plt.show()

fig = plt.figure(figsize=(15, 10))
ax = plt.gca()
df.hist(bins=20, ax=ax, layout=(3, 3), column = int_columns)
plt.tight_layout()
# plt.savefig('Plots/Distribution_num_cols_v2.png')
plt.show()


#plot the Price and log(Price) column
f, axes = plt.subplots(2, 1)
sns.histplot(x=df.loc[df['msrp']< 100000,'msrp'], bins=50, ax=axes[0])
axes[0].set_xlabel('Price')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Histogram of the price')
sns.histplot(x=np.log1p(df.loc[df['msrp']< 100000,'msrp']), bins=50, ax=axes[1])
axes[1].set_xlabel('Price')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Histogram of the log(price)')
plt.show()


#Create Train, Validation and Test set
help(train_test_split)
train_test_split()

#create X and y (features and target variables)
df_features, df_target = df.drop(columns=['msrp'], axis=1), df['msrp']

#Split the data
df_train, df_val, y_train, y_val = train_test_split(df_features, df_target, train_size=0.7, random_state=69)
df_val, df_test, y_val, y_test = train_test_split(df_val, y_val, train_size=0.5, random_state=69)

#check all the dims
list(map(lambda x: x.shape, [df_train, df_val, df_test]))
list(map(lambda x: x.shape, [y_train, y_val, y_test]))

#create log(target)
y_train_log, y_val_log, y_test_log = list(map(lambda x: np.log1p(x), [y_train, y_val, y_test]))


#Linear Regression -> raw_implementation
def train_linear_regresion(X, y):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    
    XTX = X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    return w[0], w[1:]


base_features = [x for x in int_columns if x not in ('msrp','year')]

def prep_df(df):
    X = df.fillna(0)
    return X.values

X_train = prep_df(df_train[base_features])

w0, w = train_linear_regresion(X_train, y_train_log)
y_pred = w0 + X_train.dot(w)

plt.figure(figsize=(16, 4))
sns.histplot(x=y_pred, label = 'prediction', color ='yellow', alpha=0.6, bins=40)
sns.histplot(x=y_train_log, label = 'actual', color ='green', alpha=0.8, bins=40)

plt.legend()
plt.ylabel('Frequency')
plt.xlabel('Log(1+msrp')
plt.title('Prediction vs Actual distribution of the log(msrp)')
plt.show()

#define function for RMSE
def rmse(y_pred, y_actual):
    return np.sqrt(((y_pred - y_actual) ** 2).mean())

rmse(y_pred, y_train_log) # 0.7322251711652434

w0, w = train_linear_regresion(X_train, y_train_log)
X_val = prep_df(df_val[base_features])

y_val_pred = w0 + X_val.dot(w)
rmse(y_val_pred, y_val_log) # 0.7507735305592439

#Plot the error
plt.figure(figsize=(16, 4))
sns.histplot(x=y_val_pred, label = 'prediction', color ='yellow', alpha=0.6, bins=40)
sns.histplot(x=y_val_log, label = 'actual', color ='green', alpha=0.8, bins=40)
plt.legend()
plt.ylabel('Frequency')
plt.xlabel('Log(1+msrp')
plt.title('Prediction vs Actual distribution of the log(msrp)')
plt.show()


#add age as a feature
df_train['age'] = np.max(df_train['year']) - df_train['year']


#def train pipeline, trian on train set and validate on val set
def train_pipeline(df_train, df_val, y_train, y_val, features):
    # pdb.set_trace()
    if features == 'all':
        X_train = prep_df(df_train)
    else: 
        X_train = prep_df(df_train[features])
        
    w0, w = train_linear_regresion(X_train, y_train)
    
    y_pred = w0 + X_train.dot(w)
    rmse_error = rmse(y_pred, y_train)

    print(f'Train error is: {rmse_error}')
    
    X_val = prep_df(df_val[features])
    y_pred = w0 + X_val.dot(w)
    rmse_error = rmse(y_pred, y_val)
    print(f'Validation error is: {rmse_error}')
    
    return print('[INFO] ---------- > completed!')

train_pipeline(df_train, df_val, y_train_log, y_val_log, base_features)
train_pipeline(df_train, df_val, y_train_log, y_val_log, base_features + ['year']) # just by adding the year feature we decreased the error with 0.2 

#adding some string columns 
print(string_columns)
temp_dict = {}

for c in string_columns:
    print(c)
    temp_dict[c] = list(df[c].value_counts().head().index)


add_make = [{c:v} for c,v in temp_dict.items()][0]

#Simple Feature Engineering take string column, take its first 5 values and loop trough the function 
def f_engineering(df, any_dict):
    df_new = df.copy()
    features = []
    for c, values in any_dict.items():  
        for v in values:
            df_new[f'{c}_{v}'] = (df_new[c] == v).astype(int)
            features.append(f'{c}_{v}')
      
    return df_new, features

df_new, new_features = f_engineering(df_train, add_make)        

train_pipeline(df_new, df_val, y_train_log, y_val_log, base_features + ['year'] + new_features)


#TODO FINISH THE REGULARIZATION

#check persentage NAs 
df_train.isna().mean()*100
df.iloc[(df_train.duplicated() == True).index]

df.loc[df['make'] == "Mitsubishi"][df.loc[df['make'] == "Mitsubishi"].duplicated(keep=False)]
df[df.duplicated(subset = ['make'])]

