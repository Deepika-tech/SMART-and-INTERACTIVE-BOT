#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Import and preprocess the data
df = pd.read_excel('D:/Data Engineering/Abby International.xlsx')

# Renaming columns
df.rename(columns={'Custoemr ID': 'Customer_ID', 'Created time': 'Created_time'}, inplace=True)

# Making a copy of 'Customer_ID' and 'Type'
df['Customer_ID_copy'] = df['Customer_ID']
df['Type_copy'] = df['Type']

# Convert 'Created_time' to datetime
df['Created_time'] = pd.to_datetime(df['Created_time'])

# Sort by 'Customer_ID' and 'Created_time' to ensure the sequence of interactions
df = df.sort_values(by=['Customer_ID', 'Created_time'])

# Step 2: Combine rare classes into a single "Other" category and ensure 'Type' column is string
def combine_rare_classes(df, column, min_count=2):
    while True:
        value_counts = df[column].value_counts()
        rare_classes = value_counts[value_counts < min_count].index
        if len(rare_classes) == 0:
            break
        df[column] = df[column].apply(lambda x: 'Other' if x in rare_classes else x)
    return df

df = combine_rare_classes(df, 'Type')
df['Type'] = df['Type'].astype(str)

# Check class distribution
print("Class distribution after combining rare classes:")
print(df['Type'].value_counts())

# Step 3: Encode the 'Type' and 'Customer_ID' columns
label_encoder_type = LabelEncoder()
df['Type'] = label_encoder_type.fit_transform(df['Type'].astype(str))

label_encoder_customer = LabelEncoder()
df['Customer_ID'] = label_encoder_customer.fit_transform(df['Customer_ID'].astype(str))

# Save the label encoders for future use
joblib.dump(label_encoder_type, 'D:/ML_OPS/New/label_encoder_type.pkl')
joblib.dump(label_encoder_customer, 'D:/ML_OPS/New/label_encoder_customer.pkl')

# Create lag features for sequence prediction
def create_lag_features(df, target_column, max_lag=5):
    for i in range(1, max_lag + 1):
        df[f'{target_column}_lag_{i}'] = df.groupby('Customer_ID')[target_column].shift(i)
    df.dropna(inplace=True)
    return df

# Apply lag feature creation with a maximum lag of 5
df = create_lag_features(df, target_column='Type', max_lag=5)

# Define features (X) and target (y)
X = df[[f'Type_lag_{i}' for i in range(1, 6)]]
y = df['Type']

# Ensure that the stratified split won't fail by checking the minimum count
min_count = y.value_counts().min()
print(f"Minimum class count after combining rare classes: {min_count}")

# Split the data using stratified sampling
while True:
    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        break
    except ValueError as e:
        print(f"Splitting failed: {e}")
        print("Combining more rare classes...")
        df['Type'] = df['Type_copy']  # Restore original Type values
        df = combine_rare_classes(df, 'Type', min_count=min_count+1)
        df['Type'] = label_encoder_type.fit_transform(df['Type'].astype(str))
        y = df['Type']
        min_count = y.value_counts().min()

print(f"Final class distribution in the training set:\n{pd.Series(y_train).value_counts()}")
print(f"Final class distribution in the validation set:\n{pd.Series(y_val).value_counts()}")

# Define the XGBoost model with best parameters from hyperparameter tuning
model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective='multi:softmax', num_class=len(label_encoder_type.classes_))

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, 'D:/ML_OPS/New/best_xgboost_model.pkl')

print("Model and encoders saved successfully.")

# Create and display the mapping table with predictions
label_mapping = pd.DataFrame({
    'Category': label_encoder_type.classes_,
    'Numeric Label': label_encoder_type.transform(label_encoder_type.classes_)
})

# Function to predict the next questions and their probabilities
def predict_next_questions(sequence, model, label_encoder_type, max_lag=5):
    # Ensure sequence contains only known types
    valid_sequence = [q for q in sequence if q in label_encoder_type.classes_]
    
    # Encode the sequence using the label encoder
    encoded_sequence = label_encoder_type.transform(valid_sequence)
    
    # Create a DataFrame for the input sequence with lag features
    sequence_df = pd.DataFrame([encoded_sequence], columns=[f'Type_lag_{i}' for i in range(1, len(encoded_sequence) + 1)])
    
    # Add missing lag features if the sequence is shorter than max_lag
    for i in range(len(encoded_sequence) + 1, max_lag + 1):
        sequence_df[f'Type_lag_{i}'] = -1  # Assuming -1 was used for filling NA in lag features
    
    # Predict the next question probabilities
    next_question_probs = model.predict_proba(sequence_df)[0]
    
    # Get the top predicted questions and their probabilities
    top_indices = next_question_probs.argsort()[-3:][::-1]  # Get indices of top 3 predictions
    top_questions = label_encoder_type.inverse_transform(top_indices)
    top_probabilities = next_question_probs[top_indices]
    
    return top_questions, top_probabilities

# Add columns for the next predicted questions and their probabilities
label_mapping['Next Question 1'] = None
label_mapping['Probability 1'] = None
label_mapping['Next Question 2'] = None
label_mapping['Probability 2'] = None
label_mapping['Next Question 3'] = None
label_mapping['Probability 3'] = None

# Populate the DataFrame with predictions and probabilities
for index, row in label_mapping.iterrows():
    current_question = row['Category']
    next_questions, probabilities = predict_next_questions([current_question], model, label_encoder_type)
    label_mapping.at[index, 'Next Question 1'] = next_questions[0]
    label_mapping.at[index, 'Probability 1'] = probabilities[0]
    if len(next_questions) > 1:
        label_mapping.at[index, 'Next Question 2'] = next_questions[1]
        label_mapping.at[index, 'Probability 2'] = probabilities[1]
    if len(next_questions) > 2:
        label_mapping.at[index, 'Next Question 3'] = next_questions[2]
        label_mapping.at[index, 'Probability 3'] = probabilities[2]

# Display the final table
print(label_mapping)


# In[ ]:





# # Random Forest and XG - Boosters

# In[39]:


#importing required libraries


# In[40]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# # Basic changes
# 
# Step 1: Import and preprocess the data
# - Renaming columns
# - Making a copy of 'Customer_ID' and 'Type'
# - Convert 'Created_time' to datetime
# 

# In[45]:


df = pd.read_excel('D:/Data Engineering/Abby International.xlsx')

df.rename(columns={'Custoemr ID': 'Customer_ID', 'Created time': 'Created_time'}, inplace=True)

df['Customer_ID_copy'] = df['Customer_ID']
df['Type_copy'] = df['Type']

df['Created_time'] = pd.to_datetime(df['Created_time'])


# # Processing!
# 
# - Sort by 'Customer_ID' and 'Created_time' to ensure the sequence of interactions
# - Combining rare classess

# In[46]:


df = df.sort_values(by=['Customer_ID', 'Created_time'])

# Step 2: Combine rare classes into a single "Other" category and ensure 'Type' column is string
def combine_rare_classes(df, column, min_count=2):
    while True:
        value_counts = df[column].value_counts()
        rare_classes = value_counts[value_counts < min_count].index
        if len(rare_classes) == 0:
            break
        df[column] = df[column].apply(lambda x: 'Other' if x in rare_classes else x)
    return df


# In[51]:


df.info()


# ### combining the rare classess

# In[56]:


df = combine_rare_classes(df, 'Type')
df['Type'] = df['Type'].astype(str)

# Check class distribution
print("Class distribution after combining rare classes:")
print(df['Type'].value_counts())


# ### Encode the 'Type' and 'Customer_ID' columns
# 

# In[57]:


label_encoder_type = LabelEncoder()
df['Type'] = label_encoder_type.fit_transform(df['Type'].astype(str))

label_encoder_customer = LabelEncoder()
df['Customer_ID'] = label_encoder_customer.fit_transform(df['Customer_ID'].astype(str))


# ### Saving the label encoders for future use

# In[58]:


joblib.dump(label_encoder_type, 'D:/ML_OPS/New/label_encoder_type.pkl')
joblib.dump(label_encoder_customer, 'D:/ML_OPS/New/label_encoder_customer.pkl')


# In[60]:


label_encoder_type


# In[61]:


label_encoder_customer


# ### Create lag features for sequence prediction

# In[ ]:


def create_lag_features(df, target_column, max_lag=5):
    for i in range(1, max_lag + 1):
        df[f'{target_column}_lag_{i}'] = df.groupby('Customer_ID')[target_column].shift(i)
    df.dropna(inplace=True)
    return df


# In[63]:


df.info()


# In[64]:


# Apply lag feature creation with a maximum lag of 5


# In[66]:


df = create_lag_features(df, target_column='Type', max_lag=5)


# #### Define features (X) and target (y)

# In[67]:


X = df[[f'Type_lag_{i}' for i in range(1, 6)]]
y = df['Type']

# Ensure that the stratified split won't fail by checking the minimum count
min_count = y.value_counts().min()
print(f"Minimum class count after combining rare classes: {min_count}")

# Split the data using stratified sampling
while True:
    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        break
    except ValueError as e:
        print(f"Splitting failed: {e}")
        print("Combining more rare classes...")
        df['Type'] = df['Type_copy']  # Restore original Type values
        df = combine_rare_classes(df, 'Type', min_count=min_count+1)
        df['Type'] = label_encoder_type.fit_transform(df['Type'].astype(str))
        y = df['Type']
        min_count = y.value_counts().min()


# In[68]:


print(f"Final class distribution in the training set:\n{pd.Series(y_train).value_counts()}")
print(f"Final class distribution in the validation set:\n{pd.Series(y_val).value_counts()}")


# In[69]:


### Define the XGBoost model with best parameters from hyperparameter tuning


# In[70]:


xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective='multi:softmax', num_class=len(label_encoder_type.classes_))

# Train the XGBoost model
xgb_model.fit(X_train, y_train)

# Evaluate the XGBoost model
xgb_y_pred = xgb_model.predict(X_val)
xgb_accuracy = accuracy_score(y_val, xgb_y_pred)
print(f"XGBoost Validation Accuracy: {xgb_accuracy:.2f}")


# In[71]:


# Save the XGBoost model
joblib.dump(xgb_model, 'D:/ML_OPS/New/best_xgboost_model.pkl')


# ### Define the Random Forest model

# In[73]:


rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

# Train the Random Forest model
rf_model.fit(X_train, y_train)

# Evaluate the Random Forest model
rf_y_pred = rf_model.predict(X_val)
rf_accuracy = accuracy_score(y_val, rf_y_pred)
print(f"Random Forest Validation Accuracy: {rf_accuracy:.2f}")


# In[77]:


# Save the Random Forest model
joblib.dump(rf_model, 'D:/ML_OPS/New/best_random_forest_model.pkl')
print("Models and encoders saved successfully.")


# ### Create and display the mapping table with predictions

# In[80]:


label_mapping = pd.DataFrame({
    'Category': label_encoder_type.classes_,
    'Numeric Label': label_encoder_type.transform(label_encoder_type.classes_)
})

# Function to predict the next questions and their probabilities
def predict_next_questions(sequence, model, label_encoder_type, max_lag=5):
    # Ensure sequence contains only known types
    valid_sequence = [q for q in sequence if q in label_encoder_type.classes_]
    
    # Encode the sequence using the label encoder
    encoded_sequence = label_encoder_type.transform(valid_sequence)
    
    # Create a DataFrame for the input sequence with lag features
    sequence_df = pd.DataFrame([encoded_sequence], columns=[f'Type_lag_{i}' for i in range(1, len(encoded_sequence) + 1)])
    
    # Add missing lag features if the sequence is shorter than max_lag
    for i in range(len(encoded_sequence) + 1, max_lag + 1):
        sequence_df[f'Type_lag_{i}'] = -1  # Assuming -1 was used for filling NA in lag features
    
    # Predict the next question probabilities
    next_question_probs = model.predict_proba(sequence_df)[0]
    
    # Get the top predicted questions and their probabilities
    top_indices = next_question_probs.argsort()[-3:][::-1]  # Get indices of top 3 predictions
    top_questions = label_encoder_type.inverse_transform(top_indices)
    top_probabilities = next_question_probs[top_indices]
    
    return top_questions, top_probabilities

# Add columns for the next predicted questions and their probabilities
label_mapping['Next Question 1'] = None
label_mapping['Probability 1'] = None
label_mapping['Next Question 2'] = None
label_mapping['Probability 2'] = None
label_mapping['Next Question 3'] = None
label_mapping['Probability 3'] = None

# Populate the DataFrame with predictions and probabilities
for index, row in label_mapping.iterrows():
    current_question = row['Category']
    next_questions, probabilities = predict_next_questions([current_question], xgb_model, label_encoder_type)
    label_mapping.at[index, 'Next Question 1'] = next_questions[0]
    label_mapping.at[index, 'Probability 1'] = probabilities[0]
    if len(next_questions) > 1:
        label_mapping.at[index, 'Next Question 2'] = next_questions[1]
        label_mapping.at[index, 'Probability 2'] = probabilities[1]
    if len(next_questions) > 2:
        label_mapping.at[index, 'Next Question 3'] = next_questions[2]
        label_mapping.at[index, 'Probability 3'] = probabilities[2]

# Display the final table
label_mapping


# In[ ]:





# # Final Version

# In[86]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Step 1: Import and preprocess the data
df = pd.read_excel('D:/Data Engineering/Abby International.xlsx')

# Renaming columns
df.rename(columns={'Custoemr ID': 'Customer_ID', 'Created time': 'Created_time'}, inplace=True)

# Making a copy of 'Customer_ID' and 'Type'
df['Customer_ID_copy'] = df['Customer_ID']
df['Type_copy'] = df['Type']

# Convert 'Created_time' to datetime
df['Created_time'] = pd.to_datetime(df['Created_time'])

# Sort by 'Customer_ID' and 'Created_time' to ensure the sequence of interactions
df = df.sort_values(by=['Customer_ID', 'Created_time'])



# In[87]:


# Step 2: Combine rare classes into a single "Other" category and ensure 'Type' column is string
def combine_rare_classes(df, column, min_count=2):
    while True:
        value_counts = df[column].value_counts()
        rare_classes = value_counts[value_counts < min_count].index
        if len(rare_classes) == 0:
            break
        df[column] = df[column].apply(lambda x: 'Other' if x in rare_classes else x)
    return df

df = combine_rare_classes(df, 'Type')
df['Type'] = df['Type'].astype(str)

# Check class distribution
print("Class distribution after combining rare classes:")
print(df['Type'].value_counts())


# In[89]:


# Step 3: Encode the 'Type' and 'Customer_ID' columns
label_encoder_type = LabelEncoder()
df['Type'] = label_encoder_type.fit_transform(df['Type'].astype(str))

label_encoder_customer = LabelEncoder()
df['Customer_ID'] = label_encoder_customer.fit_transform(df['Customer_ID'].astype(str))

# Create lag features for sequence prediction
def create_lag_features(df, target_column, max_lag=5):
    for i in range(1, max_lag + 1):
        df[f'{target_column}_lag_{i}'] = df.groupby('Customer_ID')[target_column].shift(i)
    df.dropna(inplace=True)
    return df


# In[90]:


# Apply lag feature creation with a maximum lag of 5
df = create_lag_features(df, target_column='Type', max_lag=5)

# Define features (X) and target (y)
X = df[[f'Type_lag_{i}' for i in range(1, 6)]]
y = df['Type']


# In[94]:


# Ensure that the stratified split won't fail by checking the minimum count
min_count = y.value_counts().min()
print(f"Minimum class count after combining rare classes: {min_count}")

# Split the data using stratified sampling
while True:
    try:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        break
    except ValueError as e:
        print(f"Splitting failed: {e}")
        print("Combining more rare classes...")
        df['Type'] = df['Type_copy']  # Restore original Type values
        df = combine_rare_classes(df, 'Type', min_count=min_count+1)
        df['Type'] = label_encoder_type.fit_transform(df['Type'].astype(str))
        y = df['Type']
        min_count = y.value_counts().min()
        
print(f"Final class distribution in the training set:\n{pd.Series(y_train).value_counts()}")
print(f"Final class distribution in the validation set:\n{pd.Series(y_val).value_counts()}")


# In[95]:


# Hyperparameter tuning for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)


# In[96]:


print(f"Final class distribution in the training set:\n{pd.Series(y_train).value_counts()}")
print(f"Final class distribution in the validation set:\n{pd.Series(y_val).value_counts()}")


# In[97]:


# Best model
best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_val)

# Evaluate the Random Forest model
rf_accuracy = accuracy_score(y_val, y_pred_rf)
rf_f1 = f1_score(y_val, y_pred_rf, average='weighted')
rf_precision = precision_score(y_val, y_pred_rf, average='weighted')
rf_recall = recall_score(y_val, y_pred_rf, average='weighted')

print(f"Random Forest Validation Accuracy: {rf_accuracy:.2f}")
print(f"Random Forest F1 Score: {rf_f1:.2f}")
print(f"Random Forest Precision: {rf_precision:.2f}")
print(f"Random Forest Recall: {rf_recall:.2f}")


# ### Define the XGBoost model with best parameters from hyperparameter tuning

# In[98]:


xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, objective='multi:softmax', num_class=len(label_encoder_type.classes_))

# Train the XGBoost model
xgb_model.fit(X_train, y_train)

# Evaluate the XGBoost model
y_pred_xgb = xgb_model.predict(X_val)
xgb_accuracy = accuracy_score(y_val, y_pred_xgb)
xgb_f1 = f1_score(y_val, y_pred_xgb, average='weighted')
xgb_precision = precision_score(y_val, y_pred_xgb, average='weighted')
xgb_recall = recall_score(y_val, y_pred_xgb, average='weighted')

print(f"XGBoost Validation Accuracy: {xgb_accuracy:.2f}")
print(f"XGBoost F1 Score: {xgb_f1:.2f}")
print(f"XGBoost Precision: {xgb_precision:.2f}")
print(f"XGBoost Recall: {xgb_recall:.2f}")


# In[99]:


# Save the models and encoders only after training and evaluation
joblib.dump(label_encoder_type, 'D:/ML_OPS/New/label_encoder_type.pkl')
joblib.dump(label_encoder_customer, 'D:/ML_OPS/New/label_encoder_customer.pkl')
joblib.dump(best_rf_model, 'D:/ML_OPS/New/best_random_forest_model.pkl')
joblib.dump(xgb_model, 'D:/ML_OPS/New/best_xgboost_model.pkl')

print("Models and encoders saved successfully.")


# In[ ]:




