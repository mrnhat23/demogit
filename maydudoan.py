import numpy as np
import pandas as pd
import random
import os


# Load data from txt file
input_file = os.path.abspath("C:/Users/Mr.Nhat/Desktop/cau.txt")
output_file = os.path.abspath("C:/Users/Mr.Nhat/Desktop/cau1.csv")

def get_result(number):
    digits_sum = sum(int(d) for d in str(number))
    return 0 if digits_sum <= 10 else 1

with open(input_file, "r", encoding="utf-16") as input_f, open(output_file, "w", encoding="utf-8") as output_f:
    for line in input_f:
        number = int(line.strip().lstrip('\ufeff'))
        result = get_result(number)
        output_f.write(str(result) + "\n")

# Load data from csv file
data = pd.read_csv('C:/Users/Mr.Nhat/Desktop/cau1.csv')

# Define the number of states and actions
num_states = 2 ** len(data.columns)
num_actions = 2

# Initialize the Q-table
q_table = np.zeros((num_states, num_actions))

# Define the epsilon-greedy exploration strategy
def epsilon_greedy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        # Explore
        action = random.randint(0, num_actions - 1)
    else:
        # Exploit
        action = np.argmax(q_table[state])
    return action

# Define the function to update the Q-table
def update_q_table(state, action, reward, next_state, alpha, gamma):
    max_q_next_state = np.max(q_table[next_state])
    q_table[state, action] = (1 - alpha) * q_table[state, action] + \
                             alpha * (reward + gamma * max_q_next_state)

# Define the function to get the current state
def get_state(history):
    state = 0
    for i in range(len(history)):
        state += int(history[i] * 2 ** i)
    return state

# Define the function to get the reward
def get_reward(action, result):
    if action == result:
        return 1
    else:
        return -1

# Train the Q-learning model
epsilon = 0.1
alpha = 0.1
gamma = 0.9

# Read data from csv file into a list
history = data.values.tolist()

# Make a prediction and add the result to the history list
current_state = get_state(history[-1])
action = epsilon_greedy(current_state, epsilon)
result = random.randint(0, 1)
reward = get_reward(action, result)
next_state = get_state(history[-1][1:] + [result])
update_q_table(current_state, action, reward, next_state, alpha, gamma)
history.append([result])

# Update the CSV file with the new result
new_data = pd.DataFrame(history, columns=data.columns)
new_data.to_csv('history.csv', index=False)

# Train the model on the updated history
for i in range(len(history) - 1):
    state = get_state(history[i])
    action = epsilon_greedy(state, epsilon)
    result = history[i+1][-1]
    reward = get_reward(action, result)
    next_state = get_state(history[i][1:] + [result])
    update_q_table(state, action, reward, next_state, alpha, gamma)

# Make a prediction for the next game
current_state = get_state(history[-1])
action = np.argmax(q_table[current_state])
result = action
history.append([result])

# Update the CSV file with the new result
new_data = pd.DataFrame(history, columns=data.columns)
new_data.to_csv('history.csv', index=False)

print("Prediction: ", result)