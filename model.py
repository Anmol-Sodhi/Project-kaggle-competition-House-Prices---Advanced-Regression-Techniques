#importing required libraries 
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load the training data
training_data_path = r'C:\Users\anmol\.kaggle\house-prices-data\train.csv'
training_data = pd.read_csv(training_data_path)

# Target variable
y = training_data.SalePrice

# Features (expanded with additional features for potential improvement)
features = [
    'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 
    'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'OverallQual', 
    'OverallCond', 'Neighborhood', 'ExterQual', 'KitchenQual', 'GrLivArea'
]

X = training_data[features]

# Handling categorical variables using One-Hot Encoding
categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundling preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# making the  models
rf_model = RandomForestRegressor(random_state=1, n_estimators=200, max_depth=15)

''' older version of xgb_model
xgb_model = XGBRegressor(random_state=1, n_estimators=100, learning_rate=0.1, max_depth=5) '''

# Newer version with better hyperparamter tuning increased the accuracy by about almost 5% from 0.15225 previosly to 0.14545  
xgb_model = XGBRegressor(
    random_state=1,
    n_estimators=3000,       # Increased number of trees
    learning_rate=0.005,     # Lower learning rate for more careful learning
    max_depth=6,            # Slightly deeper trees to capture more complexity
    min_child_weight=3,     # More conservative to prevent overfitting
    subsample=0.7,          # Use 80% of the data to grow each tree
    colsample_bytree=0.7,   # Use 80% of the features to grow each tree
    gamma=0.2               # Make splitting more conservative
)


# Bundling preprocessing and modeling code in a pipeline
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', rf_model)])
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', xgb_model)])

# Splitting the data into training and validation groups
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Fitting the models
rf_pipeline.fit(X_train, y_train)
xgb_pipeline.fit(X_train, y_train) 

# Getting predictions on validation data
rf_preds = rf_pipeline.predict(X_valid)
xgb_preds = xgb_pipeline.predict(X_valid)

# Evaluatong  the models
rf_mae = mean_absolute_error(y_valid, rf_preds)
xgb_mae = mean_absolute_error(y_valid, xgb_preds)

print(f"Random Forest MAE: {rf_mae}")
print(f"XGBoost MAE: {xgb_mae}")

# Chooseng  the model with the best performance (lower MAE)
if rf_mae < xgb_mae:
    best_model = rf_pipeline
    print("Random Forest selected.")
else:
    best_model = xgb_pipeline
    print("XGBoost selected.")

# Loading  the test data
test_data_path = r'C:\Users\anmol\.kaggle\house-prices-data\test.csv'
test_data = pd.read_csv(test_data_path)

# Applyng  the best model to the test data
test_X = test_data[features]
test_preds = best_model.predict(test_X)



# submission file
submission = pd.DataFrame({
    'Id': test_data.Id,
    'SalePrice': test_preds
})
submission_file_path = r'C:\Users\anmol\Desktop\submission.csv'
submission.to_csv(submission_file_path, index=False)


