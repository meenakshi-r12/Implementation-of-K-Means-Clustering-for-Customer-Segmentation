# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Importing essential libraries for data handling, visualization, and clustering.
2. Load the customer dataset into a pandas DataFrame.
3. Preview the dataset.
4. Use only the relevant numerical features: Annual Income and Spending Score.
5. Calculate WCSS (Within-Cluster Sum of Squares) for different cluster counts.
6. Plot the results to visually determine the "elbow point" where adding more clusters doesn’t significantly reduce WCSS.
7. Train the KMeans model with n_clusters=5.
8. Predict cluster labels for each customer.
9. Assign each data point to its respective cluster.
10. Use a scatter plot to visualize how customers are grouped based on Annual Income and Spending Score 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: 
RegisterNumber:  
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("NO. of. clusters")
plt.ylabel("wcss")
plt.title("Elbow method")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])
y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```
## Output:

![WhatsApp Image 2025-05-25 at 09 11 53_8bfded38](https://github.com/user-attachments/assets/fbd56cf1-1fd6-4d6b-b390-95f09bcfb58a)

![WhatsApp Image 2025-05-25 at 09 12 07_1b038839](https://github.com/user-attachments/assets/c40568bd-162b-470a-a939-0429776b1661)

![WhatsApp Image 2025-05-25 at 09 12 22_aa006337](https://github.com/user-attachments/assets/33ecc1ea-61e8-42e6-bb1f-9d52d0a37892)

![WhatsApp Image 2025-05-25 at 09 12 34_e7ca4a89](https://github.com/user-attachments/assets/f3067907-22e0-4f69-a251-5f8036d689ef)

![WhatsApp Image 2025-05-25 at 09 12 45_72e9d494](https://github.com/user-attachments/assets/23a2bbfa-8a1e-4450-8dee-9b05884f6b4d)

![WhatsApp Image 2025-05-25 at 09 12 59_b32bcd63](https://github.com/user-attachments/assets/64e8814f-ca9d-4cd9-8ccd-48567c569b57)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
