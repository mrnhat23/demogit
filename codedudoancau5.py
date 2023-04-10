import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import gym
from gym import spaces

# Định nghĩa môi trường 
class TaiXiuEnvironment(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2) # Tài hoặc xỉu
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,)) # 6 features đầu vào
        
        self.history = pd.read_csv('C:/Users/Mr.Nhat/Desktop/cau1.csv') # Load dữ liệu lịch sử
        self.reward_range = (-np.inf, np.inf)
        
        self.current_step = None
        self.max_step = len(self.history) - 1
        
    def step(self, action):
        # Thực hiện dự đoán kết quả
        X = self.get_features()
        y_pred = self.model.predict(X.reshape(1, -1))[0]
        
        # Cập nhật kết quả
        self.update_history(action, y_pred)
        
        # Tính reward dựa trên kết quả mới và kết quả dự đoán trước đó
        reward = self.get_reward(y_pred, action)
        
        # Kiểm tra nếu đây là bước cuối cùng của vòng lặp
        if self.current_step == self.max_step:
            done = True
        else:
            done = False
        
        # Tạo observation mới
        obs = self.get_features()
        
        # Trả về các giá trị của bước hiện tại
        return obs, reward, done, {}
    
    def reset(self):
        # Khởi tạo mô hình học tăng cường
        self.agent = LinearAgent(self.observation_space.shape[0], self.action_space.n)
        self.current_step = 0
        
        # Khởi tạo mô hình hồi quy tuyến tính
        X, y = self.get_train_data()
        self.model = LinearRegression().fit(X, y)
        
        # Trả về observation đầu tiên
        return self.get_features()
    
    def get_features(self):
        # Lấy các feature đầu vào
        last_result = self.history.loc[self.current_step, 'result']
        tai_ratio = self.history.loc[self.current_step, 'tai_ratio']
        xiu_ratio = self.history.loc[self.current_step, 'xiu_ratio']
        tai_frequency = self.history.loc[self.current_step, 'tai_frequency']
        xiu_frequency = self.history.loc[self.current_step, 'xiu_frequency']
        tai_sequence = self.history.loc[self.current_step, 'tai_sequence']
        
        # Chuyển đổi chuỗi thành các giá trị nhị phân và gộp các feature lại thành một vector
        tai_sequence = np.array([int(i) for i in list(tai_sequence)])
        tai_sequence = np.pad(tai_sequence, (5 - len(tai_sequence), 0), mode='constant') # Thêm padding cho độ dài chuỗi bằng 5
        features = np.array([last_result, tai_ratio, xiu_ratio, tai_frequency, xiu_frequency])
        features = np.concatenate((features, tai_sequence))
        
        return features

    def update_history(self, action, result):
        # Cập nhật lịch sử trò chơi
        self.history.loc[self.current_step, 'action'] = action
        self.history.loc[self.current_step, 'result'] = result
        
        # Cập nhật tỉ lệ và tần suất xuất hiện
        if result == 1: # Tài
            self.history.loc[self.current_step, 'tai_ratio'] = (self.history.loc[:self.current_step, 'result'] == 1).mean()
            self.history.loc[self.current_step, 'xiu_ratio'] = (self.history.loc[:self.current_step, 'result'] == 0).mean()
            self.history.loc[self.current_step, 'tai_frequency'] = (self.history.loc[:self.current_step, 'result'] == 1).sum()
            self.history.loc[self.current_step, 'xiu_frequency'] = (self.history.loc[:self.current_step, 'result'] == 0).sum()
        else: # Xỉu
            self.history.loc[self.current_step, 'tai_ratio'] = (self.history.loc[:self.current_step, 'result'] == 1).mean()
            self.history.loc[self.current_step, 'xiu_ratio'] = (self.history.loc[:self.current_step, 'result'] == 0).mean()
            self.history.loc[self.current_step, 'tai_frequency'] = (self.history.loc[:self.current_step, 'result'] == 1).sum()
            self.history.loc[self.current_step, 'xiu_frequency'] = (self.history.loc[:self.current_step, 'result'] == 0).sum()
            
        # Cập nhật chuỗi tài/xỉu
        tai_sequence = self.history.loc[:self.current_step, 'result'].tail(5)
        tai_sequence = ''.join([str(i) for i in tai_sequence])
        self.history.loc[self.current_step, 'tai_sequence'] = tai_sequence
        
        self.current_step += 1
        
    def get_reward(self, y_pred, action):
        # Tính reward dựa trên kết quả mới và kết quả dự đoán trước đó
        last_result = self.history.loc[self.current_step-1, 'result']
        if action == last_result:
            reward = 1
        else:
            reward = -1
        
        if y_pred == action:
            reward += 1
        else:
            reward -= 1
        
        return reward

    def get_train_data(self):
        # Lấy dữ liệu để train mô hình hồi quy tuyến tính
        X = []
        y = []
        for i in range(5, len(self.history)):
            features = self.get_features_from_history(i)
            result = self.history.loc[i, 'result']
            
            X.append(features)
            y.append(result)
            
        return np.array(X), np.array(y)

    def get_features_from_history(self, i):
        # Lấy các feature từ lịch sử trò chơi
        last_result = self.history.loc[i-1, 'result']
        tai_ratio = (self.history.loc[:i-1, 'result'] == 1).mean()
        xiu_ratio = (self.history.loc[:i-1, 'result'] == 0).mean()
        tai_frequency = (self.history.loc[:i-1, 'result'] == 1).sum()
        xiu_frequency = (self.history.loc[:i-1, 'result'] == 0).sum()
        tai_sequence = self.history.loc[i-5:i-1, 'result']
        
        # Chuyển đổi chuỗi thành các giá trị nhị phân và gộp các feature lại thành một vector
        tai_sequence = np.array([int(i) for i in tai_sequence])
        tai_sequence = np.pad(tai_sequence, (5 - len(tai_sequence), 0), mode='constant') # Thêm padding cho độ dài chuỗi bằng 5
        features = np.array([last_result, tai_ratio, xiu_ratio, tai_frequency, xiu_frequency])
        features = np.concatenate((features, tai_sequence))
        
        return features
#Định nghĩa mô hình học tăng cường
class LinearAgent:
    def init(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95 # discount rate
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        
        return model

    def act(self, state):
        # Chọn hành động dựa trên state hiện tại
        q_values = self.model.predict(state)
        action = np.argmax(q_values[0])
        return action

    def train(self, state, action, reward, next_state, done):
        # Huấn luyện mô hình học tăng cường với dữ liệu mới
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)
print("Predicted class:", prediction)