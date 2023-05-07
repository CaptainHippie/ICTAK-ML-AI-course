# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore')
pd.options.display.max_columns = 50

# %%
df_train = pd.read_csv('train_v9rqX0R.csv')
df_train.head(10)

# %%
df_test = pd.read_csv('test_AbJTz2l.csv')
df_test.head(10)

# %%
df_train.info()

# %%
df_train.nunique()

# %%
df_train.isna().sum()

# %%
df_train.corr(numeric_only=True)

# %% [markdown]
# # Preprocess Outlets Data

# %% [markdown]
# Minmax Scale Outlet_Establishment_Year
# 
# onehot encode Outlet_Identifier
# 
# create new column: is_supermarket
# 
# onehot encode supermarket types 1, 2, 3
# 
# label encode Outlet_Location_Type : Tier 1,2,3
# 
# label encode outlet size (note: has missing values)
# 
# Use Machine Learning to predict missing values for Outlet_Size

# %%
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# %%
minmax = MinMaxScaler(feature_range=(0,1))
le = LabelEncoder()

# %% [markdown]
# creating new dataframe containing only features related to the Outlet

# %%
outlet_df = df_train.iloc[:,6:-1].copy()
outlet_df

# %% [markdown]
# Defining necessary functions in one place

# %%
def swap_sizes(x):
    if x==1:
        return 3
    elif x == 3:
        return 1
    elif x == 2:
        return 2
    else:
        return None

# %%
def encode_outlet_types(dataframe):
    dataframe['is_supermarket'] = ((dataframe['Outlet_Type'] != 'Grocery Store')).astype(int)
    dataframe['SM_type1'] = ((dataframe['Outlet_Type'] == 'Supermarket Type1')).astype(int)
    dataframe['SM_type2'] = ((dataframe['Outlet_Type'] == 'Supermarket Type2')).astype(int)
    dataframe['SM_type3'] = ((dataframe['Outlet_Type'] == 'Supermarket Type3')).astype(int)
    dataframe.drop(columns=['Outlet_Type'], inplace=True)

def minmax_scale_year(dataframe):
    dataframe['Outlet_Establishment_Year'] = minmax.fit_transform(dataframe[['Outlet_Establishment_Year']])

def onehot_encode_outlet_identifier(dataframe):
    dataframe['OI'] = dataframe['Outlet_Identifier']
    dataframe = pd.get_dummies(dataframe, columns = ['OI'])
    dataframe.drop(columns=['Outlet_Identifier'], inplace=True)
    return dataframe

def label_encode_outlet_size(dataframe):
    dataframe['Outlet_Size'] = le.fit_transform(dataframe['Outlet_Size']) + 1
    #dataframe['Outlet_Size'] = dataframe['Outlet_Size'].apply(swap_sizes)
    return dataframe

def label_encode_outlet_location_type(dataframe):
    dataframe['Outlet_Location_Type'] = le.fit_transform(dataframe['Outlet_Location_Type']) + 1

# %%
def fill_missing_outlet_types(dataframe, predicted_values):
    index = 0
    for i in range(0, len(dataframe)):
        item = dataframe['Outlet_Size'][i]
        if pd.isna(item):
            dataframe['Outlet_Size'][i] = predicted_values[index]
            index += 1

# %%
# creates new is_supermarket column and onehot encodes the supermarket types
encode_outlet_types(outlet_df)

# minmax scales the established year column from 0 to 1
minmax_scale_year(outlet_df)

# onehot encodes the unique outlet identifier columns (10 new columns)
outlet_df = onehot_encode_outlet_identifier(outlet_df)

# encoding Outlet location types Tier 1, Tier2, Tier3 as 1,2 & 3 respectively
label_encode_outlet_location_type(outlet_df)

# encoding Outlet sizes small, medium, High as 1,2 & 3 respectively
outlet_df = label_encode_outlet_size(outlet_df)
outlet_df['Outlet_Size'] = outlet_df['Outlet_Size'].apply(swap_sizes)

# %%
outlet_df

# %% [markdown]
# #### Creating model to predict Outlet Sizes

# %% [markdown]
# creating testing and training datasets

# %%
outlet_train = outlet_df[outlet_df['Outlet_Size'].notna()].copy()
outlet_size_to_predict = outlet_df[outlet_df['Outlet_Size'].isna()].copy()
outlet_size_input_to_predict = outlet_size_to_predict.drop(columns=['Outlet_Size'])

# %%
x_outlet = outlet_train.drop(columns=['Outlet_Size'])
y_outlet = outlet_train['Outlet_Size']

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# %%
x_train_outlet, x_val_outlet, x_test_outlet, y_val_outlet = train_test_split(x_outlet, y_outlet, test_size=0.35, random_state=134)

# %%
lr_out = LogisticRegression()
lr_out.fit(x_train_outlet, x_test_outlet)
y_val_pred_out = lr_out.predict(x_val_outlet)
accuracy_score(y_val_outlet, y_val_pred_out)

# %% [markdown]
# Model gives a perfect accuracy score of 1

# %%
predicted_outlet_sizes = lr_out.predict(outlet_size_input_to_predict)
predicted_outlet_sizes

# %% [markdown]
# Fill the missing outlet sizes in outlet_df sequentially and convert from float to integer column

# %%
fill_missing_outlet_types(outlet_df, predicted_outlet_sizes)
outlet_df['Outlet_Size'] = outlet_df['Outlet_Size'].astype('int')

# %% [markdown]
# Fully processed Outlet data

# %%
outlet_df

# %% [markdown]
# ### Creating outlet dataset for testing set

# %%
outlet_df_test = df_test.iloc[:,6:].copy()
outlet_df_test

# %% [markdown]
# Applying the same functions previously applied on the training set

# %%
encode_outlet_types(outlet_df_test)
minmax_scale_year(outlet_df_test)
outlet_df_test = onehot_encode_outlet_identifier(outlet_df_test)
label_encode_outlet_location_type(outlet_df_test)
outlet_df_test = label_encode_outlet_size(outlet_df_test)
outlet_df_test['Outlet_Size'] = outlet_df_test['Outlet_Size'].apply(swap_sizes)

# %% [markdown]
# Filling missing outlet_sizes. From the observations in the training dataset, the outlet_sizes for the outlet identifiers are: OUT045 = 1, OUT017 = 1, OUT010 = 2

# %%
outlet_df_test.loc[outlet_df_test['OI_OUT010']==1, 'Outlet_Size'] = 2
outlet_df_test.loc[outlet_df_test['OI_OUT017']==1, 'Outlet_Size'] = 1
outlet_df_test.loc[outlet_df_test['OI_OUT045']==1, 'Outlet_Size'] = 1
outlet_df_test['Outlet_Size'] = outlet_df_test['Outlet_Size'].astype('int')

# %% [markdown]
# Fully processed outlet testing dataset

# %%
outlet_df_test

# %% [markdown]
# # Preprocess Items Data

# %% [markdown]
# Label encode Item_Fat_Content low fat and regular fat as 0 and 1 respectively
# 
# onehot encode Item_Type
# 
# Fill the missing values in Item_Weight. The same Item_Identifier should have the same Item_Weight
# 
# Remove outliers in Item_weight, Item_Visibility and Item_MRP (only when training)

# %%
items_df = df_train.iloc[:,:6].copy()
items_df

# %% [markdown]
# Save corresponding item weights for each item identifier as a dictionary

# %%
item_weight_mappings = {}

for i in range(0, len(items_df)):
    item = items_df['Item_Identifier'][i]
    weight = items_df['Item_Weight'][i]
    if not pd.isna(weight):
        item_weight_mappings[item] = weight

# %% [markdown]
# Define necessary functions in one place

# %%
def encode_item_fat_content(dataframe):
    dataframe['Item_Fat_Content'] = ((dataframe['Item_Fat_Content']=='Regular') | (dataframe['Item_Fat_Content']=='reg')).astype(int)

def fill_missing_item_types(dataframe):
    for i in range(0, len(dataframe)):
        item = dataframe['Item_Identifier'][i]
        weight = dataframe['Item_Weight'][i]
        if pd.isna(weight):
            if item in item_weight_mappings.keys():
                dataframe['Item_Weight'][i] = item_weight_mappings[item]

def onehot_encode_item_types(dataframe):
    dataframe = pd.get_dummies(dataframe, columns = ['Item_Type'])
    return dataframe

# %%
len(list(item_weight_mappings.keys()))

# %% [markdown]
# There are four mappings missing. So the identifier does not have a default weight to assign to

# %%
# Label encode Item_Fat_Content low fat and regular fat as 0 and 1 respectively
encode_item_fat_content(items_df)

# Fill the missing values in Item_Weight based on the mapping created earlier
fill_missing_item_types(items_df)

# onehot encoding Item_Types
items_df = onehot_encode_item_types(items_df)

# %% [markdown]
# To deal with the four identifiers with no default weight

# %%
unique_item_weights = (items_df.groupby(['Item_Identifier'])['Item_Weight'].nunique() == 0).to_dict()
have_no_weight_to_assign = []
for key in unique_item_weights:
    if unique_item_weights[key] == True:
        have_no_weight_to_assign.append(key)
have_no_weight_to_assign

# %%
items_with_no_weight = items_df[(items_df['Item_Identifier'] == 'FDE52') | (items_df['Item_Identifier'] == 'FDK57') | (items_df['Item_Identifier'] == 'FDN52') | (items_df['Item_Identifier'] == 'FDQ60')]
items_with_no_weight

# %% [markdown]
# Filling the four missing rows with the average weight of their corresponding Item type. Also adding them to the weight mapping

# %%
frozed_food_avg = round(items_df.loc[items_df['Item_Type_Frozen Foods'] == 1, "Item_Weight"].mean(), 3)
snack_foods_avg = round(items_df.loc[items_df['Item_Type_Snack Foods'] == 1, "Item_Weight"].mean(), 3)
dairy_avg = round(items_df.loc[items_df['Item_Type_Dairy'] == 1, "Item_Weight"].mean(), 3)
baking_goods_avg = round(items_df.loc[items_df['Item_Type_Baking Goods'] == 1, "Item_Weight"].mean(), 3)

items_df['Item_Weight'][927] = frozed_food_avg
items_df['Item_Weight'][1922] = snack_foods_avg
items_df['Item_Weight'][4187] = dairy_avg
items_df['Item_Weight'][5022] = baking_goods_avg

item_weight_mappings['FDN52'] = frozed_food_avg
item_weight_mappings['FDK57'] = snack_foods_avg
item_weight_mappings['FDE52'] = dairy_avg
item_weight_mappings['FDQ60'] = baking_goods_avg

# %% [markdown]
# Fully processed items training dataset

# %%
items_df

# %% [markdown]
# ### Preparing testing dataset

# %%
items_df_test = df_test.iloc[:,:6].copy()

# %% [markdown]
# Applying the same functions we did in the training dataset

# %%
encode_item_fat_content(items_df_test)
fill_missing_item_types(items_df_test)
items_df_test = onehot_encode_item_types(items_df_test)

# %%
items_df_test

# %%
print(items_df.isna().sum().any())
print(items_df_test.isna().sum().any())

# %% [markdown]
# # Merging Items and Outlet datasets

# %%
final_train_df = pd.concat((items_df, outlet_df), axis=1)
final_train_df

# %%
final_test_df = pd.concat((items_df_test, outlet_df_test), axis=1)
final_test_df

# %% [markdown]
# ### Feature Selection based on correlation

# %%
X_train_df = final_train_df.iloc[:, [1,2,3,4,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]].copy()
X_test_df = final_test_df.iloc[:, [1,2,3,4,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37]].copy()
Y_train_all = df_train['Item_Outlet_Sales']

# %%
X_train_df.head()

# %% [markdown]
# Minmax scaling some numeric columns

# %%
X_train_df[['Item_Weight','Item_MRP','Item_Visibility']] = minmax.fit_transform(X_train_df[['Item_Weight','Item_MRP','Item_Visibility']])
X_test_df[['Item_Weight','Item_MRP','Item_Visibility']] = minmax.transform(X_test_df[['Item_Weight','Item_MRP','Item_Visibility']])

# %%
X_train, X_val, y_train, y_val = train_test_split(X_train_df, Y_train_all, test_size=0.2, random_state=13)

# %% [markdown]
# # Model Training and testing on testing set itself

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# %% [markdown]
# ### Using GradientBoostingRegressor Model

# %% [markdown]
# All Hyperparams : min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_depth, max_leaf_nodes, max_features, learning_rate, n_estimators, subsample, loss, init
# 
# We'll select just 3

# %%
param_grid = {
    'learning_rate': [0.15, 0.16, 0.17, 0.18],
    'max_depth' : [2,3,4],
    'n_estimators':[35, 40, 45, 50, 55]
} 
gbr_gscv = GradientBoostingRegressor()
gbr_gscv = GridSearchCV(gbr_gscv, param_grid=param_grid, n_jobs=1, scoring='neg_mean_absolute_error', cv=5, verbose=2)
gbr_gscv.fit(X_train, y_train)

# %%
print(gbr_gscv.best_params_)
print(gbr_gscv.best_score_)

# %% [markdown]
# The best hyperparameters are learning_rate=0.16, max_depth=3, n_estimators=36

# %%
Gbr = GradientBoostingRegressor(learning_rate=0.15, max_depth=3, n_estimators=35)
Gbr.fit(X_train, y_train)
y_pred_gbr = Gbr.predict(X_val)

# %%
print("Mean Absolute Error :", mean_absolute_error(y_val, y_pred_gbr))
print("Root Mean Squared Error :", mean_absolute_error(y_val, y_pred_gbr))

# %% [markdown]
# ### Training using Entire training dataset

# %%
Final_model = GradientBoostingRegressor(learning_rate=0.15, max_depth=3, n_estimators=35)
Final_model.fit(X_train_df, Y_train_all)

# %% [markdown]
# ## Making final Predictions

# %%
final_predictions = Final_model.predict(X_test_df)

# %%
df_sol = pd.read_csv('sample_submission_8RXa3c6.csv')
df_sol.head()

# %%
df_submission = df_sol.copy()

# %%
df_submission['Item_Outlet_Sales'] = final_predictions
df_submission

# %% [markdown]
# Checking if there's any negative or zero values

# %%
(df_submission['Item_Outlet_Sales']<=0).any()

# %% [markdown]
# Exporting file to CSV

# %%
df_submission.to_csv('final_submission.csv', index=False)


