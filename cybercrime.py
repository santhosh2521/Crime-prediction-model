import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame
df = pd.read_csv("cybercrimedata.csv", encoding='ISO-8859-1')

# Convert 'Method' and 'Organization_type' columns to lowercase
df['Method'] = df['Method'].str.lower()
df['Organization_type'] = df['Organization_type'].str.lower()

# Combine 'hacked' and 'Hacked' into a single category
df['Method'] = df['Method'].replace({'hacked': 'hack'})

# Count the occurrences of each attack method and industry attacked
method_counts = df['Method'].value_counts()
industry_counts = df['Organization_type'].value_counts()

# Set up a matplotlib figure with subplots and increased spacing
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 12), gridspec_kw={'hspace': 0.56})

# Plot the most common attack methods
sns.barplot(x=method_counts.head(10).index, y=method_counts.head(10).values, ax=axes[0])
axes[0].set_title('Top 10 Most Common Attack Methods')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)

# Plot the most common industries attacked with increased x-coordinate spacing
sns.barplot(x=industry_counts.head(10).index, y=industry_counts.head(10).values, ax=axes[1])
axes[1].set_title('Top 10 Most Common Industries Attacked')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_xticklabels(axes[1].get_xticklabels(), ha="right")  # Align x-labels to the right

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
