import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


modelnb = GaussianNB()
pizzaDataset = pd.read_csv('Pizza.csv',names = ['Brand', 'mois', 'prot', 'fat', 'ash','sodium','carb','cal'], header=0, sep=",")
x = pizzaDataset.drop(["Brand"], axis=1)
x.head(300)
y = pizzaDataset["Brand"]
y.head(300)
x_test = pizzaDataset.drop(["Brand"], axis=1)
x_test.head(300)
y_uji = pizzaDataset["Brand"]
y_uji.head(300)
nbtrain = modelnb.fit(x, y)
mois = input("mois ? \n" + ">>>")
prot = input("prot ? \n" + ">>>")
fat = input("fat ? \n" + ">>>")
ash = input("ash ? \n" + ">>>")
sodium = input("sodium ? \n" + ">>>")
carb = input("carb ? \n" + ">>>")
cal = input("cal ? \n" + ">>>")
moisData = float(mois)
protData = float(prot)
fatData = float(fat)
ashData = float(ash)
sodiumData = float(sodium)
carbData = float(carb)
calData = float(cal)
prediksinya = modelnb.predict([[moisData, protData, fatData, ashData, sodiumData, carbData, calData ]])

print(prediksinya)