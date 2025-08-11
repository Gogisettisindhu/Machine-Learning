import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df=pd.read_csv('Lung Cancer.csv')
x=df[['age','bmi','cholesterol_level','asthma']]
y=df['survived']
#print(df.info)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
res = r2_score(y_test, y_pred)
print("R2 Score:",res)

# Take new input
a = int(input('Enter age:'))
c = int(input('enter bmi:'))
d = int(input('Enter cholesterol_level:'))
e = int(input('asthma:'))
ans = model.predict([[a,c, d, e]])
print(ans)
