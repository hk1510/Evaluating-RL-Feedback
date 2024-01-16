import pandas as pd
import numpy as np
import seaborn as sns
import json
from numpy import array, float32
import matplotlib.pyplot as plt

# d =' {"speed": 3.0650431315104156, "crashed": "False", "action": "array([-0.9867545, -0.770202 ], dtype=float32)", "is_success": "True"}'
# d = dict(d)
# print(d)

# train_set = [1, 3, 5]
# test_set = [17, 19, 21]
# train_groups = {0: [], 1: [], 2: [], 3: []}
# test_groups = {0: [], 1: [], 2: [], 3: []}

train_set = [0, 2, 4]
practice_set = [6, 8, 10, 12, 14]
test_set = [16, 18, 20]
data = [[], [], [], []]

train_groups = {0: np.zeros([4, 3]), 1: np.zeros([4, 3]), 2: np.zeros([4, 3]), 3: np.zeros([4, 3])}
practice_groups = {0: np.zeros([4, 5]), 1: np.zeros([4, 5]), 2: np.zeros([4, 5]), 3: np.zeros([4, 5])}
test_groups = {0: np.zeros([4, 3]), 1: np.zeros([4, 3]), 2: np.zeros([4, 3]), 3: np.zeros([4, 3])}


df = pd.read_csv('results_new_seeds.csv', header=None)


train_idx = np.zeros(4, dtype=np.int64)
practice_idx = np.zeros(4, dtype=np.int64)
test_idx = np.zeros(4, dtype=np.int64)
for index, row in df.iterrows():
    group = row[22]
    data[group].append([])
    # get crash data for the training rounds
    for i in range(len(train_set)):
        info = float(row[train_set[i]])
        train_groups[group][train_idx[group]][i] += info
        data[group][-1].append(info)
        # train_groups[group].append(int(info["crashed"]))
    # get the data for the practice rounds
    for i in range(len(practice_set)):
        info = float(row[practice_set[i]])
        practice_groups[group][practice_idx[group]][i] += info
        data[group][-1].append(info)
        
        # train_groups[group].append(int(info["crashed"]))
    # get crash data for the testing rounds
    for i in range(len(test_set)):
        info = float(row[test_set[i]])
        test_groups[group][test_idx[group]][i] += info
        data[group][-1].append(info)
        
        # test_groups[group].append(int(info["crashed"]))
    train_idx[group] += 1
    practice_idx[group] += 1
    test_idx[group] += 1

print(np.array(data))
with open('data.txt', 'w') as f:
    f.write(str(np.array(data)))


raise Exception

# get the average crashes for each group
# for k in train_groups.keys():
#     train_groups[k] = np.mean(train_groups[k])
#     test_groups[k] = np.mean(test_groups[k])

data = {0: [], 1: [], 2: [], 3: []}
data = [[], [], []]
for participant in range(len(train_groups[0])):
    for i in range(3):
        print(train_groups[0][participant][i])
        # data = np.append(data, (i + 1, train_groups[0][participant][i]))
        data[i].append(train_groups[0][participant][i])
print(data)
data = np.array(data)

# get the average of each group 
# for j in range(4):
#     practice_groups[j] = (practice_groups[j]) / 5
#     train_groups[j] = (train_groups[j]) / 3
#     test_groups[j] = (test_groups[j]) / 3
#     print(train_groups[j], practice_groups[j])
#     data[j] = np.concatenate([train_groups[j], practice_groups[j], test_groups[j]], axis=None)


print(data)
sns.lineplot(data = train_groups, errorbar=("ci", 95))

# plt.savefig("results.png")

plt.show()

# data = pd.Series(train_groups, name = "Numerical Variable")
# print(data)
# sns.histplot(data=data, discrete=True)
# data = []
# groups = ["Control", "Group A", "Group B", "Group C"]
# for i in range(0, len(train_groups.values())):
#     data.append([groups[i], 'Test', test_groups[i]])
#     data.append([groups[i], 'Train', train_groups[i]])
    



# df = pd.DataFrame(data, columns=['group','Round','val'])
# df.pivot(columns = "Round",  index= "group", values="val").plot(kind='bar')
# plt.xlabel("Group")
# plt.ylabel("Average Number of Crashes")
# plt.subplots_adjust(bottom=0.15)
# plt.xticks(rotation = 0)
# plt.title('Average Number of Crashes Per Round')


# plt.savefig("Average Number of Crashes Per Round.png")

# plt.show()



# plt.bar(train_groups.keys(), train_groups.values())

# Plot the responses for different events and regions
# sns.lineplot(x="timepoint", y="signal",
#              hue="region", style="event",
#              data=fmri, errorbar=("ci", 95))

# plt.savefig("95.png") 
# plt.show()

# sns.lineplot(x="timepoint", y="signal",
#              hue="region", style="event",
#              data=fmri, errorbar=("ci", 80))

# plt.savefig("80.png") 
