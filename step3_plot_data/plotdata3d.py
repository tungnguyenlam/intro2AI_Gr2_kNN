import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data= pd.read_csv('updated_pollution_dataset.csv')
x = data['SO2']
y = data['CO']
z = data['Proximity_to_Industrial_Areas']
t = data['Air Quality']

category_map = {'Good': 0, 'Moderate': 1, 'Poor': 2, 'Hazardous': 3}
t = t.map(category_map)

#encode labels to numbers


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
ax.grid()
colors= ['green','yellow', 'blue', 'purple']
labels= list(category_map.keys())

for label, color in zip(category_map.values(), colors):
    data_subset= data[t==label]
    ax.scatter(data_subset['SO2'], data_subset['CO'], data_subset['Proximity_to_Industrial_Areas'], c=color, label=labels[label], s=50)

# Set axes label
ax.set_xlabel('SO2')
ax.set_ylabel('CO')
ax.set_zlabel('Proximity')
ax.legend()

plt.show()