#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

#fetching datasets
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
stores = pd.read_csv('stores.csv')
features = pd.read_csv('features.csv')

# Convert date columns to datetime
for df in [train, test, features]:
    df["Date"] = pd.to_datetime(df["Date"])

# Merge features and stores into train/test
train = train.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
train = train.merge(stores, on="Store", how="left")

test = test.merge(features, on=["Store", "Date", "IsHoliday"], how="left")
test = test.merge(stores, on="Store", how="left")

#handling missing values
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
train[markdown_cols] = train[markdown_cols].fillna(0)
test[markdown_cols] = test[markdown_cols].fillna(0)

# Forward fill CPI and Unemployment
train[["CPI", "Unemployment"]] = train[["CPI", "Unemployment"]].ffill()
test[["CPI", "Unemployment"]] = test[["CPI", "Unemployment"]].ffill()

#feature engineering
def create_time_features(df):
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Day"] = df["Date"].dt.day
    df["IsMonthStart"] = df["Date"].dt.is_month_start.astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
    return df

train = create_time_features(train)
test = create_time_features(test)

# Sort by Store, Dept, Date
train = train.sort_values(by=["Store", "Dept", "Date"])

# Lag Features (1 week)
train["Weekly_Sales_Lag_1"] = train.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1)

# Rolling Mean (last 4 weeks)
train["Rolling_Mean_4"] = train.groupby(["Store", "Dept"])["Weekly_Sales"].shift(1).rolling(window=4).mean()

train = train.dropna()

# Feature columns
features_cols = [
    "Store", "Dept", "Size", "Temperature", "Fuel_Price",
    "CPI", "Unemployment", "IsHoliday",
    "MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5",
    "Year", "Month", "Week", "IsMonthStart", "IsMonthEnd",
    "Weekly_Sales_Lag_1", "Rolling_Mean_4"
]
X = train[features_cols]
y = train["Weekly_Sales"]

# Split by date (train before July 2012, validate after)
cutoff_date = pd.to_datetime("2012-07-01")
X_train = X[train["Date"] < cutoff_date]
y_train = y[train["Date"] < cutoff_date]

X_valid = X[train["Date"] >= cutoff_date]
y_valid = y[train["Date"] >= cutoff_date]

# Linear Regression (baseline)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_valid)

# LightGBM
lgb_model = lgb.LGBMRegressor()
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_valid)

#evaluation
rmse_lr = mean_squared_error(y_valid, y_pred_lr) ** 0.5
rmse_lgb = mean_squared_error(y_valid, y_pred_lgb) ** 0.5

print("RMSE:", rmse_lr)
print("LightGBM RMSE:", rmse_lgb)

# Sum by week
valid_dates = train[train["Date"] >= cutoff_date]["Date"]
df_plot = pd.DataFrame({
    "Date": valid_dates.values,
    "Actual": y_valid.values,
    "Predicted": y_pred_lgb
})
agg = df_plot.groupby("Date").sum()

#plotting
agg.plot(figsize=(12, 6), title="Actual vs. Predicted Sales (Overall)")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.grid()
plt.show()

# Fill missing lag/rolling values in test
test["Weekly_Sales_Lag_1"] = np.nan
test["Rolling_Mean_4"] = np.nan  # not enough data to compute these reliably in test

# Use same feature set (with lag features filled as 0)
test[["Weekly_Sales_Lag_1", "Rolling_Mean_4"]] = test[["Weekly_Sales_Lag_1", "Rolling_Mean_4"]].fillna(0)

X_test = test[features_cols]
test_preds = lgb_model.predict(X_test)

# Create submission DataFrame
submission = test.copy()
submission["Id"] = submission["Store"].astype(str) + "_" + submission["Dept"].astype(str) + "_" + submission["Date"].dt.strftime('%Y-%m-%d')
submission["Weekly_Sales"] = test_preds

submission = submission[["Id", "Weekly_Sales"]]
submission.to_csv("sales_forecast_submission.csv", index=False)
print("Submission file saved as 'sales_forecast_submission.csv'")

