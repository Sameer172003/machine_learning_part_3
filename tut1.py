# Decision Tree (Classification) Practical

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset=pd.read_csv("C:\\Users\\ojhas\\Dropbox\\PC\\Downloads\\Social_Network_Ads.csv")
# print(dataset.head())

x=dataset.iloc[:,:-1]
y=dataset["Purchased"]

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(x)
x=pd.DataFrame(sc.transform(x),columns=x.columns)
# print(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

from sklearn.tree import DecisionTreeClassifier
dt1=DecisionTreeClassifier()
dt1.fit(x_train,y_train)
print(dt1.score(x_test,y_test)*100)

print()

dt2=DecisionTreeClassifier(criterion="entropy")
dt2.fit(x_train,y_train)
print(dt2.score(x_test,y_test)*100)

from sklearn.tree import plot_tree
plt.figure(figsize=(50,50))
plot_tree(dt2)
plt.savefig("demo.jpg")
plt.show()