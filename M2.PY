import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv('data.csv')

# Features and target
x = df[['temp','hum','ph']]
y = df['rain']

# Split data correctly
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(x_train,y_train)

# Predict
y_pred = model.predict(x_test)

# Evaluate
res = r2_score(y_test, y_pred)
print("R2 Score:",res)

# Take new input
a = int(input('Enter temp:'))
b = int(input('Enter hum:'))
c = int(input('Enter ph:'))

ans = model.predict([[a, b, c]])
print("Predicted Rainfall:",ans[0])