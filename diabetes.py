import pandas as pd
df=pd.read_csv('diabetes.csv')
print(df.info)
print(df.head(10))
print(df.tail())
print(df.shape)
print(df.describe())
print('before droping',df.shape)
df=df.drop_duplicates()
print('after droping')
df.isnull().sum()
print('no of missing values in glucose',df[df['Glucose']==0].shape[0])
print('no of missing values in BMI',df[df['BMI']==0].shape[0])
print('no of missing values in BloodPressure',df[df['BloodPressure']==0].shape[0])
print('no of missing values in Pregnancies',df[df['Pregnancies']==0].shape[0])
print('no of missing values in SkinThickness',df[df['SkinThickness']==0].shape[0])
print('no of missing values in Insulin',df[df['Insulin']==0].shape[0]) 
print('no of missing values in Age',df[df['Age']==0].shape[0])

df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())
df['BMI']=df['BMI'].replace(0,df['BMI'].mean())
df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())
df['Pregnancies']=df['Pregnancies'].replace(0,df['Pregnancies'].mean())
df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].mean())
df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())
df['Age']=df['Age'].replace(0,df['Age'].mean())
print('no of missing values in glucose',df[df['Glucose']==0].shape[0])
print('no of missing values in BMI',df[df['BMI']==0].shape[0])
print('no of missing values in BloodPressure',df[df['BloodPressure']==0].shape[0])
print('no of missing values in Pregnancies',df[df['Pregnancies']==0].shape[0])
print('no of missing values in SkinThickness',df[df['SkinThickness']==0].shape[0])
print('no of missing values in Insulin',df[df['Insulin']==0].shape[0]) 
print('no of missing values in Age',df[df['Age']==0].shape[0])

print(df.describe())
x=df.drop(columns='Outcome',axis=1)
#x=df['Age','BloodPressure','BMI']
y=df['Outcome']
print(x.head())
print(y.head())
print(df.head())


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x)
sd=scaler.transform(x)
x=sd
y=df.Outcome
print(x)
print(y)
