import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data= pd.read_csv('country_wise_latest.csv')
attribute = ['Deaths / 100 Cases', 'Recovered / 100 Cases', 'Active', 'WHO Region']
x = data[attribute[0]]
y = data[attribute[1]]
z = data[attribute[2]]
t = data[attribute[3]]

category_map = { 'Europe': 0, 'Africa': 1, 'Americas': 2, 'Western Pacific': 3}
t = t.map(category_map)

#encode labels to numbers


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(projection='3d')
ax.grid()
colors= ['green','yellow', 'blue', 'purple', 'red', 'orange']
labels= list(category_map.keys())

for label, color in zip(category_map.values(), colors):
    data_subset= data[t==label]
    ax.scatter(data_subset[attribute[0]], data_subset[attribute[1]], data_subset[attribute[2]], c=color, label=labels[label], s=50)

# Set axes label
ax.set_xlabel(attribute[0])
ax.set_ylabel(attribute[1])
ax.set_zlabel(attribute[2])
ax.legend()
plt.title("Covid-19 cases based on Confirmed last week, 1 week change and 1 week % increase number of cases infected")
plt.savefig('plot', dpi=300, bbox_inches='tight')
plt.show()