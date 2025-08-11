import pandas as pd 
a={'temp':[32,34,36,28,44,56,18,33,36,10],
   'hum':[64,52,56,87,35,65,18,56,54],
   'ph':[1,2,3,14,15,3,4,6,7,8],
   'rain':[32,45,13,45,33,56,76,57,89,79]}
df = pd.DataFrame(a)
print(df) 
df.to_csv('data.csv',index s= False) 