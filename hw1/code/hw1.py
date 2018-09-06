import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

all_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combined_data = [all_df, test_df]
all_df = pd.concat(combined_data)

def header():
    print('-'*66)
    return
# Q7
header()
print(all_df.describe())
header()
# Q8
print(all_df.describe(include=['O']))
header()
# Q9
# print(
#     all_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# )

# Q10
# print(
#     all_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# )

# Q11
# graph_age = sns.FacetGrid(all_df, col='Survived')
# graph_age.map(plt.hist, 'Age', bins=20)
# plt.show()

# Q12
# graph_pclass = sns.FacetGrid(all_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# graph_pclass.map(plt.hist, 'Age', alpha=.5, bins=20)
# graph_pclass.add_legend()
# plt.show()

# Q13
# graph_embarked = sns.FacetGrid(all_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
# graph_embarked.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
# graph_embarked.add_legend()
# plt.show()

# Q14
print(all_df.duplicated(subset=['Ticket'], keep='first'))
header()

# Q16
all_df['Sex'] = np.where(all_df['Sex'] == 'male', 0, 1)
print(all_df)
header()

