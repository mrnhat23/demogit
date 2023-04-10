import tkinter as tk
from tkinter import filedialog

import pandas as pd
import numpy as np
import os
import joblib

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.select_file_button = tk.Button(self, text="Select CSV file", command=self.select_csv_file)
        self.select_file_button.pack(side="top")

        self.select_model_button = tk.Button(self, text="Select model file", command=self.select_model_file)
        self.select_model_button.pack(side="top")

        self.run_button = tk.Button(self, text="Run", command=self.run_model)
        self.run_button.pack(side="bottom")

        # Add a Label widget to display the prediction result
        self.result_label = tk.Label(self, text="")
        self.result_label.pack(side="bottom")

    def select_csv_file(self):
        self.csv_file_path = filedialog.askopenfilename(initialdir="/", title="Select CSV file", filetypes=[("CSV files", "*.csv")])
        print("Selected CSV file:", self.csv_file_path)

    def select_model_file(self):
        self.model_file_path = filedialog.askopenfilename(initialdir="/", title="Select model file", filetypes=[("Pickle files", "*.pkl")])
        print("Selected model file:", self.model_file_path)

    def run_model(self):
        # Load data from csv file
        df = pd.read_csv(self.csv_file_path, header=None, names=['state', 'reward'])

        # Load Q table from pickle file
        Q = joblib.load(self.model_file_path)

        # Perform Q-learning algorithm
        alpha = 0.1
        gamma = 0.9
        epsilon = 0.1
        correct_predictions = 0
        incorrect_predictions = 0
        for i in range(1000):
            state = 0
            done = False
            while not done:
                if np.random.rand() < epsilon:
                    action = np.random.randint(2)
                else:
                    action = np.argmax(Q[state])

                next_state = state + 1
                reward = df.loc[state, 'reward']
                done = (state == len(df) - 1)

                if next_state >= len(df):
                    next_state = len(df) - 1
                    Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

                state = next_state

            # Predict the next game result
            current_state = len(df) - 1
            if np.argmax(Q[current_state]) == 0:
                prediction = 'tài'
            else:
                prediction = 'xỉu'

            # Check if the prediction is correct
            if prediction == df.loc[current_state, 'state']:
                correct_predictions += 1
            else:
                incorrect_predictions += 1
            
        # Update the text of the result label with the prediction and the number of correct and incorrect predictions
        self.result_label.configure(text="Kết quả dự đoán cho cây tiếp theo là: " + prediction + "\nSố cây dự đoán đúng: " + str(correct_predictions) + "\nSố cây dự đoán sai: " + str(incorrect_predictions))

        # Save the updated Q table to the pickle file
        joblib.dump(Q, self.model_file_path)


root = tk.Tk()
app = Application(master=root)
app.mainloop()
