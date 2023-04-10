import pandas as pd
import numpy as np
import os
import joblib

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

# Đọc dữ liệu từ file csv vào dataframe
df = pd.read_csv('C:/Users/Mr.Nhat/Desktop/cau1.csv', header=None, names=['state', 'reward'])

# Khởi tạo bảng Q với kích thước bằng số lượng trạng thái và số lượng hành động
num_states = len(df)  # Số lượng trạng thái bằng số lượng dòng trong file csv
num_actions = 2  # Số lượng hành động là 2 (Tài hoặc Xỉu)
Q = np.zeros((num_states, num_actions))  # Bảng Q ban đầu toàn giá trị 0


# Thực hiện thuật toán Q-learning
alpha = 0.1  # Tỷ lệ học
gamma = 0.9  # Hệ số chiết khấu
epsilon = 0.1  # Tỷ lệ tham lam
for i in range(1000):  # Số lần lặp
    # Khởi tạo trạng thái ban đầu là 0
    state = 0
    done = False
    while not done:
        # Chọn hành động theo chính sách epsilon-greedy
        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(Q[state])
            
        # Thực hiện hành động và quan sát kết quả
        next_state = state + 1
        reward = df.loc[state, 'reward']
        done = (state == num_states - 1)  # Kết thúc khi đạt trạng thái cuối cùng
        
        # Cập nhật bảng Q theo công thức Q-learning
        if next_state >= num_states:
            next_state = num_states - 1
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

        
        # Chuyển sang trạng thái tiếp theo
        state = next_state
# Dự đoán kết quả game tiếp theo
current_state = len(df) - 1  # Trạng thái hiện tại là trạng thái cuối cùng trong file csv
if np.argmax(Q[current_state]) == 0:
    prediction = 'tài'
else:
    prediction = 'xỉu'
print("Kết quả dự đoán cho cây tiếp theo là : ", prediction)

