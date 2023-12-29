import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

df = pd.read_csv("C:/Users/recep/Desktop/Python/Data.csv")
x = df.iloc[:,:-1].values
#print(x)
y = df.iloc[:,-1].values
#print(y)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy="mean")
imputer.fit(x[:,1:])
x[:,1:] = imputer.transform(x[:,1:])
#print(x)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

cf= ColumnTransformer(transformers=[("encoder", OneHotEncoder(),[0])],remainder= "passthrough")
x = np.array(cf.fit_transform(x))

#y = np.array(cf.fit_transform(y))
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2 ,random_state=1)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train[:,3:] = ss.fit_transform(x_train[:,3:])
x_test[:,3:] = ss.fit_transform(x_test[:,3:])

tecrube = pd.read_csv("C:/Users/recep/Desktop/Python/deneyim-maas.csv", sep = ";")
mean_column2 = tecrube['maas'].mean()
tecrube['maas'].fillna(mean_column2, inplace=True)
#plt.scatter(tecrube.deneyim, tecrube.maas)
plt.title("Maaş Tecrübe İlişkisi")
plt.xlabel("Deneyim Yılı")
plt.ylabel("Maaş Skalası")

z = tecrube.iloc[:,:-1].values
t = tecrube.iloc[:,-1].values

z_train, z_test, t_train, t_test = train_test_split(z,t,test_size=0.2 ,random_state=1)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(z_train,t_train)

t_predict = lr.predict(z_test)

plt.scatter(z_train, t_train, color = "red")
plt.plot(z_train, lr.predict(z_train) )
plt.title("Tecrube")
plt.xlabel("Deneyim Süresi")
plt.ylabel("Maaş Skalası")
plt.show()





