import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
import secrets as sec

all_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
df_list = [all_df, test_df]
all_df = pd.concat(df_list)

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_rows', 1400)

def header():
    print('-'*66)
    return

header()

# Q7
print(all_df.describe())
header()

# Q8
print(all_df.describe(include=['O']))
header()

# Q9
print(
    all_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
)
header()

# Q10
print(
    all_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
)
header()

# Q11
graph_age = sns.FacetGrid(all_df, col='Survived')
graph_age.map(plt.hist, 'Age', bins=20)
plt.show()

# Q12
graph_pclass = sns.FacetGrid(all_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
graph_pclass.map(plt.hist, 'Age', alpha=.5, bins=20)
graph_pclass.add_legend()
plt.show()

# Q13
graph_embarked = sns.FacetGrid(all_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
graph_embarked.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
graph_embarked.add_legend()
plt.show()

# Q14
# I just used simple math with values gleaned from the categorical description
header()

# Q16
all_df = all_df.rename(columns={'Sex':'Gender'})
all_df['Gender'] = np.where(all_df['Gender'] == 'male', 0, 1)
# print(all_df)
header()

# Q17
def randomGen():
    return sec.choice(range(14, 29))

for i, r in all_df.iterrows():
    if (pd.isnull(r['Age'])):
        all_df.loc[i, 'Age'] = randomGen()
# print(all_df)

# Q18
most_freq_port = all_df.Embarked.dropna().mode()[0]

all_df['Embarked'] = all_df['Embarked'].fillna(most_freq_port)
# print(all_df)

# Q19
all_df['Fare'].fillna(all_df['Fare'].dropna().mode(), inplace=True)
# print(all_df)

# Q20
all_df.loc[all_df['Fare'] <= 7.91, 'Fare'] = 0
all_df.loc[(all_df['Fare'] > 7.91) & (all_df['Fare'] <= 14.454), 'Fare'] = 1
all_df.loc[(all_df['Fare'] > 14.454) & (all_df['Fare'] <= 31), 'Fare']   = 2
all_df.loc[all_df['Fare'] > 31, 'Fare'] = 3
# print(all_df)


