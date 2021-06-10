import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

pizzaDataset = pd.read_csv('Pizza.csv',names = ['Brand', 'mois', 'prot', 'fat', 'ash','sodium','carb','cal'], header=0, sep=",")
plt.figure(1) # n adalah nomor berbeda untuk setiap window gambar
sns.scatterplot(x='mois', y='carb', hue='Brand', data=pizzaDataset).set_title("pizza by carb and mois")

plt.figure(2)
sns.scatterplot(x='fat', y='ash', hue='Brand', data=pizzaDataset).set_title("pizza by fat and ash")

plt.show()
