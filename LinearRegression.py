import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Read the Excel data sheet
data = pd.read_excel("employee_burnout_analysis-AI 2.xlsx")

# Drop rows with missing values
data = data.dropna()

# Drop 'Employee ID' column (as it has no correlation with Burn Rate)
data = data.drop('Employee ID', axis=1)

# Convert the 'Date of Joining' to days since '2008-01-01'
yeardata = pd.to_datetime("2008-01-01")
data["Days"] = (pd.to_datetime(data['Date of Joining']) - yeardata).dt.days

# Calculate correlation between numeric columns and 'Burn Rate'
numeric_data = data.select_dtypes(include=['number'])
correlation = numeric_data.corr()['Burn Rate']
print(correlation)

# Drop 'Date of Joining' and 'Days' columns as their correlation with 'Burn Rate' is small
data = data.drop(['Date of Joining', 'Days'], axis=1)

# Plotting count plots for string type columns
String_columns = data.select_dtypes(include=['object']).columns
fig, ax = plt.subplots(nrows=1, ncols=len(String_columns), sharey=True, figsize=(10, 5))
for i, c in enumerate(String_columns):
    sb.countplot(x=c, data=data, ax=ax[i])
plt.show()

# Apply dummies for categorical variables
if all(col in data.columns for col in ['Company Type', 'WFH Setup Available', 'Gender']):
    data = pd.get_dummies(data, columns=['Company Type', 'WFH Setup Available', 'Gender'], drop_first=True)

# Preprocessing: Train-test split
y = data['Burn Rate']
X = data.drop('Burn Rate', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=55)

# Scaling the data
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

# Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)

# Print the predictions
print(y_pred)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display the evaluation metrics
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")
