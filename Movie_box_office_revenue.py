import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
# from xgboost import XGBRegressor
# df=pd.read_csv("box_office.csv")
df0=pd.read_csv("boxoffice.csv")
df0=df0.drop(columns=["lifetime_gross","rank","title"])
df0
# df=pd.read_csv("box_office.csv")
# df=pd.read_csv("boxoffice.csv")
df1=pd.read_csv("merged_dataset.csv")
df1=df1.drop(columns=["movie_rated", "run_length","release_date","num_reviews"])
df1
df2 = pd.read_csv("movies_metadata.csv")

df2['release_date'] = pd.to_datetime(df2['release_date'], errors='coerce')

df2['release_month'] = df2['release_date'].dt.strftime('%B')
df2['year'] = df2['release_date'].dt.strftime('%Y')
# df2["year"]=df2["release_date"]
df3=df2[['budget', 'original_language', 'release_date', 'revenue', 'release_month','year']]
df3
# # df_=pd.merge(df0,df1)
# # # df_.to_csv("movie_.csv")
# # # df=pd.merge(df_,df3)
# # df_

# # Merge first two DataFrames on the common column
# df_temp = pd.merge(df0, df1, on='common_column', how='inner')  # Use 'left', 'right', or 'outer' as needed

# # Merge the third DataFrame with the result
# df_final = pd.merge(df_temp, df2, on='common_column', how='inner')

# # Display merged DataFrame
# print(df_final.head())



# df_temp=pd.merge(df0, df1)
# df_t=pd.merge(df_temp,df3)
# df_t




df1['year'] = pd.to_numeric(df1['year'], errors='coerce')
df3['year'] = pd.to_numeric(df3['year'], errors='coerce')

df1['year'] = df1['year'].astype(str)
df3['year'] = df3['year'].astype(str)

import pandas as pd

# Convert 'year' to the same type (choose int or str)
df1['year'] = pd.to_numeric(df1['year'], errors='coerce')  # Convert to int
df3['year'] = pd.to_numeric(df3['year'], errors='coerce')  # Convert to int

# Merge df1 (movie details) with df3 (financial data) on 'year'
df_temp = pd.merge(df1, df3, on='year', how='inner')

# If df0 has no common column, use concat
df = pd.concat([df_temp, df0], axis=1)  

df

# Save the merged dataset
# df_final.to_csv("merged_movies_dataset.csv", index=False)

df = df.head(2000)
df.isnull().sum()
df.nunique()
df.dropna(axis=0)

df['original_language'] = df['original_language'].fillna('en')
df['studio'] = df['studio'].fillna('fox')
df['year'] = df['year'].fillna('1999')
df.dropna(subset=['budget'], inplace=True)
df.isnull().sum()
df=df.drop(columns=["year", "release_date"])
df.head()
from sklearn.preprocessing import OrdinalEncoder

month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
oe = OrdinalEncoder(categories=[month_order])
df['release_month'] = oe.fit_transform(df[['release_month']])
df

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
df['studio'] = le.fit_transform(df['studio'])
df

scaler = StandardScaler()
df[['budget', 'rating', 'num_raters']] = scaler.fit_transform(df[['budget', 'rating', 'num_raters']])
X = df.drop(columns=['revenue'])
y = df['revenue']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
models_performance = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Performance:")
    print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}\n")
    models_performance[name] = {'r2_score': r2_score(y_test, y_pred), 'rmse': mean_squared_error(y_test, y_pred), 'mae': mean_absolute_error(y_test, y_pred)}
performance_df = pd.DataFrame(models_performance).T
performance_df
from sklearn.linear_model import LinearRegression

# lr = LinearRegression()
# lr.fit(X_train, y_train)

from scipy.sparse import csr_matrix
X_train_sparse = csr_matrix(X_train)  # Convert to sparse format

lr = LinearRegression()
lr.fit(X_train_sparse, y_train)
from sklearn.metrics import mean_squared_error, r2_score

y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-Squared Score (R²): {r2:.2f}")
from sklearn.linear_model import Lasso
lasso_reg = Lasso(alpha=50, max_iter=100, tol=0.1)
lasso_reg.fit(X_train, y_train)
lasso_reg.score(X_test, y_test)
lasso_reg.score(X_train, y_train)
from sklearn.linear_model import Ridge
ridge_reg= Ridge(alpha=50, max_iter=100, tol=0.1)
ridge_reg.fit(X_train, y_train)
ridge_reg.score(X_test, y_test)
ridge_reg.score(X_train, y_train)
from sklearn.linear_model import ElasticNet
elasticnet_reg= ElasticNet(alpha=0.001, max_iter=100, tol=0.1)
elasticnet_reg.fit(X_train, y_train)
elasticnet_reg.score(X_test, y_test)
elasticnet_reg.score(X_train, y_train)
