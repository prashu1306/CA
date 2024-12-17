import matplotlib.pyplot as plt
import numpy as np
# List of file names
file_names = ['plotting_ac.txt', 'plotting_ca.txt','plotting_single_ac.txt', 'plotting_dqn.txt', 'plotting_ppoac.txt', 'plotting_ppoca.txt']
file_names_sdt = ['plotting_ac_sdt.txt', 'plotting_ca_std.txt', 'plotting_single_ac_sdt.txt','plotting_dqn_sdt.txt', 'plotting_ppoac_sdt.txt', 'plotting_ppoca_sdt.txt']

# Initialize lists to store data
x_values = [i for i in range(1, 5001)]
y_values = [[] for i in range(len(file_names))]
y_values_sdt = [[] for i in range(len(file_names_sdt))]
print("Length of x_values: ", len(x_values))
print("X values: ", x_values[:5])

# Read data from each file
for i, file_name in enumerate(file_names):
    with open(file_name, 'r') as file:
        for line in file:
            # print(line)
            # Assuming each line contains x and y values separated by a space
            y_values[i].append(float(line))
        print("Length of y_values[{i}]: ", len(y_values[i]))
    file.close()

for i, file_name in enumerate(file_names_sdt):
    with open(file_name, 'r') as file:
        for line in file:
            # print(line)
            # Assuming each line contains x and y values separated by a space
            y_values_sdt[i].append(float(line))
        print("Length of y_values_sdt[{i}]: ", len(y_values_sdt[i]))
    file.close()


print('Lenght of y_values: ', len(y_values))


for i in range(len(file_names)):
    print("Lenght of y-{i}", len(y_values[i]))

labels = ['Actor-Critic', 'Critic-Actor','Single Timescale AC', 'DQN', 'PPO Actor Critic', 'PPO Critic Actor' ]

labels_sdt = ['Actor-Critic SD', 'Critic-Actor SD', 'DQN SD', 'PPO Actor Critic SD', 'PPO Critic Actor SD']

# Plot the data
for i in range(len(file_names)):
    '''if(i == 3 or i == 4):
        y1 = np.array(y_values[i])
        y2 = np.array(y_values_sdt[i])
        plt.plot(x_values[:49000], y_values[i], label=labels[i], linestyle='dashed')
        plt.fill_between(range(len(x_values[:49000])) , y1-y2,y1+y2,alpha=0.2 )
    else:'''
    y1 = np.array(y_values[i])
    y2 = np.array(y_values_sdt[i])
    plt.plot(x_values[:10000], y_values[i][:10000], label=labels[i])
    plt.fill_between(x_values[:10000],y1[:10000]-y2[:10000],y1[:10000]+y2[:10000] , alpha = 0.2)
plt.xlabel('No of Iterations')
plt.ylabel('Average reward')
plt.title('MountainCarContinuous-v0')
plt.legend()
plt.savefig('plots/avg_rewards_all.png')
plt.show()
