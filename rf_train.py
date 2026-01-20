import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score , mean_absolute_error


df = pd.read_csv("insurance.csv")

X = df.drop("charges", axis = 1)
y = df["charges"]

num_features = ["age" , "bmi" , "children" ]
cat_features = ["sex", "smoker", "region"]

#further scalling in pipeline for num_features

num_transformer = Pipeline(
    steps=[
        ('scaler', StandardScaler())
    ]
)

#further encoding in pipeline for cat_features
cat_transformer = Pipeline(
    steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]
)

## as random forest will be used so no need to handle outlier now


preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ]
)

X_train , X_test, y_train , y_test = train_test_split(
    X, y , test_size = 0.2 , random_state = 42
)


rf_best = RandomForestRegressor(n_estimators=200, max_depth= 10 , min_samples_split=5 , random_state=42, n_jobs= -1)

rf_best_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('regressor', rf_best)
    ]
)

rf_best_pipeline.fit(X_train, y_train)

y_pred = rf_best_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test , y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test , y_pred)


print(f"Mean absolution error: {mae}")
print(f"Mean squared error: {mse}")
print(f"Root mean squared error: {rmse}")
print(f"R2 score: {r2}")

with open("insurance_rf_pipeline.pkl", "wb") as file:
    pickle.dump(rf_best_pipeline, file)