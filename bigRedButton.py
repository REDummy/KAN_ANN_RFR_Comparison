import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import csv
import psutil
import time
import threading
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from kan import*
from datetime import datetime

# Check CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class SystemMonitor:
    def measure_resources(self, duration):
        process = psutil.Process()
        mem_usage, cpu_usage = [], []

        def monitor():
            start_time = time.time()
            while time.time() - start_time < duration:
                mem_info = process.memory_full_info()
                mem_usage.append(mem_info.uss / (1024 ** 2))  # MB
                cpu_usage.append(process.cpu_percent(interval=None))
                time.sleep(0.1)

        thread = threading.Thread(target=monitor)
        thread.start()
        thread.join()
        return np.mean(mem_usage), np.max(mem_usage), np.mean(cpu_usage)

class ANN(nn.Module):
    def __init__(self, input_size, architecture, activation):
        super(ANN, self).__init__()
        layers = []
        full_architecture = [input_size] + list(architecture)
        for i in range(len(full_architecture) - 1):
            layers.append(nn.Linear(full_architecture[i], full_architecture[i + 1]))
            if i < len(full_architecture) - 2:
                if activation == "relu":
                    layers.append(nn.ReLU())
                elif activation == "leaky_relu":
                    layers.append(nn.LeakyReLU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class ModelTester:
    def __init__(self, data_path, target_column, categorical_columns, test_size=0.2, n_iter=5):
        self.data_path = data_path
        self.target_column = target_column
        self.categorical_columns = categorical_columns
        self.test_size = test_size
        self.n_iter = n_iter
        self.results = []
        self.monitor = SystemMonitor()
        self.load_data()

    def load_data(self):
        # Auto-detect delimiter
        with open(self.data_path, 'r', newline='', encoding='utf-8') as file:
            sample = file.read(1024)
            delimiter = csv.Sniffer().sniff(sample).delimiter

        # Load data
        data = pd.read_csv(self.data_path, delimiter=delimiter)
        data.dropna(inplace=True)
        data[self.target_column] = np.log1p(data[self.target_column])  # Apply log transformation

        # Encode categorical columns
        self.label_encoders = {}
        for col in self.categorical_columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            self.label_encoders[col] = le

        self.X = data.drop(columns=[self.target_column]).values
        self.y = data[self.target_column].values.reshape(-1, 1)

    def run_experiment(self, scenario, model_name, params):
        print(f"Running {model_name} - Scenario {scenario} for {self.n_iter} iterations...")
        for run in range(self.n_iter):
            print(f"Iteration {run + 1}/{self.n_iter}")
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size)
            scaler = StandardScaler()
            X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
            X_train, X_test = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(X_test, dtype=torch.float32).to(device)
            y_train, y_test = torch.tensor(y_train, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

            start_time = time.time()
            avg_mem, max_mem, avg_cpu = self.monitor.measure_resources(5)

            if model_name == "ANN":
                model = ANN(X_train.shape[1], params['architecture'], params['activation']).to(device)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
                for epoch in range(params['epochs']):
                    model.train()
                    optimizer.zero_grad()
                    predictions = model(X_train)
                    loss = criterion(predictions, y_train)
                    loss.backward()
                    optimizer.step()
                model.eval()
                with torch.no_grad():
                    train_preds = model(X_train).cpu().numpy()
                    test_preds = model(X_test).cpu().numpy()
            elif model_name == "RFR":
                model = RandomForestRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], max_features=params['max_features'], random_state=42)
                model.fit(X_train.cpu().numpy(), y_train.cpu().numpy().ravel())
                train_preds, test_preds = model.predict(X_train.cpu().numpy()), model.predict(X_test.cpu().numpy())
            elif model_name == "KAN":
                dataset = {
                    'train_input': X_train.to(device),
                    'test_input': X_test.to(device),
                    'train_label': y_train.to(device),
                    'test_label': y_test.to(device)
                }
                model = KAN(width=params['architecture'], grid=params['grid'], k=params['spline_order'], device=device)
                model.fit(dataset, opt=params['optimizer'], steps=20)
                train_preds = model(dataset['train_input'])[:, 0].cpu().detach().numpy()
                test_preds = model(dataset['test_input'])[:, 0].cpu().detach().numpy()

                # Symbolic regression (optional) - only for the first run
                if run == 0:
                    lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'tan', 'abs']
                    model.auto_symbolic(lib=lib)
                    formula = model.symbolic_formula()[0][0]
                    inter_formula = ex_round(formula, 4)

                    # Save formula to a text file
                    with open(f"symbolic_formula_scenario_{scenario}.txt", "w") as file:
                        file.write(f"Scenario: {scenario}\n")
                        file.write(f"Symbolic Formula: {inter_formula}\n")

            train_mse = mean_squared_error(y_train.cpu().numpy(), train_preds)
            test_mse = mean_squared_error(y_test.cpu().numpy(), test_preds)
            train_mae = mean_absolute_error(y_train.cpu().numpy(), train_preds)
            test_mae = mean_absolute_error(y_test.cpu().numpy(), test_preds)

            self.results.append({
                "Scenario": scenario,
                "Model": model_name,
                "Time Taken (s)": time.time() - start_time,
                "Average Memory Usage (MB)": avg_mem,
                "Peak Memory Usage (MB)": max_mem,
                "Average CPU Usage (%)": avg_cpu,
                "Train MSE": train_mse,
                "Test MSE": test_mse,
                "Train MAE": train_mae,
                "Test MAE": test_mae
            })

    def run_all_experiments(self, scenarios):
        for model_name, param_list in scenarios.items():
            for i, scenario in enumerate(param_list, 1):
                self.run_experiment(i, model_name, scenario)
        self.save_results()

    def save_results(self):
        df = pd.DataFrame(self.results)
        avg_per_scenario = df.groupby(["Scenario", "Model"]).mean().reset_index()
        avg_overall = df.groupby("Model").mean().reset_index()

        today = datetime.now().strftime('%Y-%m-%d_%H-%M')

        with pd.ExcelWriter(f'model_performance_results_{today}.xlsx') as writer:
            df.to_excel(writer, sheet_name='All Results', index=False)
            avg_per_scenario.to_excel(writer, sheet_name='Average Per Scenario', index=False)
            avg_overall.to_excel(writer, sheet_name='Overall Averages', index=False)
        print("Final results saved to model_performance_results.xlsx")


if __name__ == "__main__":
    scenarios = {
        "KAN": [
            {"architecture": [1, 5, 1], "grid": 5, "spline_order": 3, "optimizer": "LBFGS"},
            {"architecture": [2, 10, 5, 1], "grid": 10, "spline_order": 3, "optimizer": "LBFGS"},
            {"architecture": [10, 20, 10, 1], "grid": 15, "spline_order": 4, "optimizer": "LBFGS"},
            {"architecture": [2, 10, 5, 1], "grid": 10, "spline_order": 5, "optimizer": "LBFGS"},
            {"architecture": [1, 5, 1], "grid": 5, "spline_order": 5, "optimizer": "LBFGS"},

        ],
        "ANN": [
            {"architecture": (1,5,1), "epochs": 100, "activation": "relu", "optimizer": "adam",
             "learning_rate": 0.01},
            {"architecture": (2, 10, 5, 1), "epochs": 200, "activation": "relu", "optimizer": "adam",
             "learning_rate": 0.01},
            {"architecture": (10, 20, 10, 1), "epochs": 300, "activation": "relu", "optimizer": "adam",
             "learning_rate": 0.01},
            {"architecture": (2, 10, 5, 1), "epochs": 200, "activation": "leaky_relu", "optimizer": "adam",
             "learning_rate": 0.01},
            {"architecture": (1, 5, 1), "epochs": 100, "activation": "leaky_relu", "optimizer": "adam",
             "learning_rate": 0.01}
        ],
        "RFR": [
            {"n_estimators": 100, "max_depth": 10, "max_features": "sqrt"},
            {"n_estimators": 100, "max_depth": 15, "max_features": "sqrt"},
            {"n_estimators": 200, "max_depth": 20, "max_features": "log2"},
            {"n_estimators": 100, "max_depth": 10, "max_features": "sqrt"},
            {"n_estimators": 50, "max_depth": 5, "max_features": "sqrt"}
        ]
    }

    tester = ModelTester("data/student-mat.csv", target_column="G3",
                         categorical_columns=["school","sex","address","famsize","Pstatus","Mjob","Fjob","reason","guardian",
                                            "schoolsup","famsup","paid","activities","nursery","higher","internet","romantic"])
    tester.run_all_experiments(scenarios)
