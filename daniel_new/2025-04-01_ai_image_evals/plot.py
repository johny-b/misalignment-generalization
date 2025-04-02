import seaborn as sns
import matplotlib.pyplot as plt

groups = ['text', 'text-image', 'comics']

data = [0.1875, 0.78125, 0.75]

# Create the plot and get the axes object
ax = sns.barplot(x=groups, y=data, hue=groups)

plt.xlabel('Modality')
plt.ylabel('Probability of expressing opinion')
plt.ylim(0, 1)
# Convert values to percentages and add labels above bars
for i, v in enumerate(data):
    percentage = f'{v*100:.1f}%'
    ax.text(i, v, percentage, ha='center', va='bottom')

