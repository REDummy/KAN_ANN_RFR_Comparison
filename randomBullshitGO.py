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
from kan import *
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
        return np.mean(mem_usage), np.sum(mem_usage), np.mean(cpu_usage)

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
    def __init__(self, datasets, test_size=0.2, n_iter=5):
        self.datasets = datasets
        self.test_size = test_size
        self.n_iter = n_iter
        self.results = []
        self.monitor = SystemMonitor()

    def load_data(self, data_path, target_column, categorical_columns=None):
        with open(data_path, 'r', newline='', encoding='utf-8') as file:
            sample = file.read(1024)
            delimiter = csv.Sniffer().sniff(sample).delimiter

        data = pd.read_csv(data_path, delimiter=delimiter)
        print(data.head())
        print(f"Before dropping NaNs: {data.isna().sum().sum()} NaN values detected.")
        data.dropna(inplace=True)
        print(f"After dropping NaNs: {data.isna().sum().sum()} NaN values detected.")

        if (data[target_column] < 0).any():
            print(f"Warning: {target_column} contains negative values, which may or may not fuck it up")
        data[target_column] = np.log1p(data[target_column])

        if categorical_columns:
            label_encoders = {}
            for col in categorical_columns:
                if col in data.columns:
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col])
                    label_encoders[col] = le

        X = data.drop(columns=[target_column]).values
        y = data[target_column].values.reshape(-1, 1)

        return X, y

    def run_experiment(self, scenario, model_name, params, dataset_name, X, y):
        print(f"Running {model_name} - {dataset_name} - Scenario {scenario} for {self.n_iter} iterations...")
        for run in range(self.n_iter):
            print(f"Iteration {run + 1}/{self.n_iter}")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
            scaler = StandardScaler()
            X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
            X_train, X_test = torch.tensor(X_train, dtype=torch.float32).to(device), torch.tensor(X_test, dtype=torch.float32).to(device)
            y_train, y_test = torch.tensor(y_train, dtype=torch.float32).to(device), torch.tensor(y_test, dtype=torch.float32).to(device)

            start_time = time.time()
            avg_mem, total_mem, avg_cpu = self.monitor.measure_resources(5)

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

                if np.isnan(train_preds).any() or np.isnan(test_preds).any():
                    print("Warning: KAN is outputting NaN predictions!")

                if run == 0:
                    lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'tan', 'abs']
                    model.auto_symbolic(lib=lib)
                    formula = model.symbolic_formula()[0][0]
                    inter_formula = ex_round(formula, 4)

                    # Save formula to a text file
                    with open(f"symbolic_formula_{dataset_name}_scenario_{scenario}.txt", "w") as file:
                        file.write(f"Scenario: {scenario}\n")
                        file.write(f"Symbolic Formula: {inter_formula}\n")

                df_sample = pd.DataFrame(X_test.cpu().numpy(), columns=[f'Feature_{i}' for i in range(X_test.shape[1])])
                df_sample['Actual'] = y_test.cpu().numpy()
                df_sample['Predicted'] = test_preds
                df_sample.to_csv(f'sample_predictions_{dataset_name}_{model_name}.csv', index=False)

            train_mse = mean_squared_error(y_train.cpu().numpy(), train_preds)
            test_mse = mean_squared_error(y_test.cpu().numpy(), test_preds)
            train_mae = mean_absolute_error(y_train.cpu().numpy(), train_preds)
            test_mae = mean_absolute_error(y_test.cpu().numpy(), test_preds)

            self.results.append({
                "Dataset": dataset_name,
                "Scenario": scenario,
                "Model": model_name,
                "Time Taken (s)": time.time() - start_time,
                "Avg Memory Usage (MB)": avg_mem,
                "Total Memory Usage (MB)": total_mem,
                "Avg CPU Usage (%)": avg_cpu,
                "Train MSE": train_mse,
                "Test MSE": test_mse,
                "Train MAE": train_mae,
                "Test MAE": test_mae
            })

    def run_all_experiments(self, scenarios):
        for dataset in self.datasets:
            X, y = self.load_data(dataset['path'], dataset['target_column'], dataset.get('categorical_columns', []))
            for model_name, param_list in scenarios.items():
                for i, scenario in enumerate(param_list, 1):
                    self.run_experiment(i, model_name, scenario, dataset['name'], X, y)
        self.save_results()

    def save_results(self):
        df = pd.DataFrame(self.results)
        avg_per_scenario = df.groupby(["Dataset", "Scenario", "Model"]).mean().reset_index()
        avg_overall = df.groupby(["Dataset", "Model"]).mean().reset_index()

        today = datetime.now().strftime('%Y-%m-%d_%H-%M')
        filename = f'model_performance_results_{today}.xlsx'

        with pd.ExcelWriter(filename) as writer:
            df.to_excel(writer, sheet_name='All Results', index=False)
            avg_per_scenario.to_excel(writer, sheet_name='Average Per Scenario', index=False)
            avg_overall.to_excel(writer, sheet_name='Overall Averages', index=False)
        print(f"Final results saved to {filename}")


if __name__ == "__main__":
    datasets = [
        {"name": "Forest Fires", "path": "data/Forest Fire/forestfires.csv", "target_column": "area",
         "categorical_columns": ["month", "day"]},
        {"name": "Student Performance", "path": "data/Student/student-mat.csv", "target_column": "G3",
         "categorical_columns": ["school","sex","address","famsize","Pstatus","Mjob","Fjob","reason","guardian",
                                            "schoolsup","famsup","paid","activities","nursery","higher","internet","romantic"]},
        {"name": "Superconductor", "path": "data/Superconductor/train.csv", "target_column": "critical_temp"},
    ]

    tester = ModelTester(datasets)

    # Extract features count **before running experiments**
    feature_counts = {}
    for dataset in datasets:
        X, _ = tester.load_data(dataset['path'], dataset['target_column'], dataset.get('categorical_columns', []))
        feature_counts[dataset['name']] = X.shape[1]

    # Generate scenarios dynamically based on extracted feature counts
    scenarios = {}
    for dataset in datasets:
        num_features = feature_counts[dataset['name']]

        scenarios[dataset['name']] = {
            "KAN": [
                {"architecture": [num_features, 5, 1], "grid": 5, "spline_order": 3, "optimizer": "LBFGS"},
                {"architecture": [num_features, 10, 5, 1], "grid": 10, "spline_order": 3, "optimizer": "LBFGS"},
                {"architecture": [num_features, 20, 10, 1], "grid": 15, "spline_order": 3, "optimizer": "LBFGS"},
                {"architecture": [num_features, 10, 5, 1], "grid": 10, "spline_order": 4, "optimizer": "LBFGS"},
                {"architecture": [num_features, 5, 1], "grid": 5, "spline_order": 4, "optimizer": "LBFGS"},
            ],
            "ANN": [
                {"architecture": (num_features, 5, 1), "epochs": 100, "activation": "relu", "optimizer": "adam",
                 "learning_rate": 0.01},
                {"architecture": (num_features, 10, 5, 1), "epochs": 200, "activation": "relu", "optimizer": "adam",
                 "learning_rate": 0.01},
                {"architecture": (num_features, 20, 10, 1), "epochs": 300, "activation": "relu", "optimizer": "adam",
                 "learning_rate": 0.01},
                {"architecture": (num_features, 10, 5, 1), "epochs": 200, "activation": "leaky_relu", "optimizer": "adam",
                 "learning_rate": 0.01},
                {"architecture": (num_features, 5, 1), "epochs": 100, "activation": "leaky_relu", "optimizer": "adam",
                 "learning_rate": 0.01}
            ],
            "RFR": [
                {"n_estimators": 100, "max_depth": 10, "max_features": "sqrt"},
                {"n_estimators": 100, "max_depth": 15, "max_features": "sqrt"},
                {"n_estimators": 200, "max_depth": 20, "max_features": "sqrt"},
                {"n_estimators": 100, "max_depth": 15, "max_features": "log2"},
                {"n_estimators": 100, "max_depth": 10, "max_features": "log2"}
            ]
        }

    # Run experiments with correct scenarios
    for dataset in datasets:
        tester.run_all_experiments(scenarios[dataset['name']])

