import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import psutil
import tkinter as tk

# Modelo de Red Neuronal Simple
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        return self.fc(x)

# Entrenamiento del modelo
def train_model():
    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    X = torch.tensor([[i] for i in range(10)], dtype=torch.float32)
    y = torch.tensor([np.random.randint(20, 80) for _ in range(10)], dtype=torch.float32).view(-1, 1)

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    return model

# Predicción de carga
def predict_load(model, task_number):
    with torch.no_grad():
        task_tensor = torch.tensor([[task_number]], dtype=torch.float32)
        return model(task_tensor).item()

# Gestión de recursos
def allocate_resources(predicted_load):
    if predicted_load > 50:
        print("Asignando recursos adicionales.")
    else:
        print("Los recursos actuales son suficientes.")

# Simulación de tareas
def simulate_task(task_id):
    print(f"Simulando tarea {task_id}...")
    # Simular carga de trabajo

# Monitorización de recursos
def monitor_resources():
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    print(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")

# Interfaz gráfica
def create_interface(model):
    def start():
        print("Inicio")
        for task_id in range(1, 6):
            simulate_task(task_id)
            predicted_load = predict_load(model, task_id)
            print(f"Predicted Load for Task {task_id}: {predicted_load:.2f}")
            allocate_resources(predicted_load)
            monitor_resources()

    def stop():
        print("Detenido")

    root = tk.Tk()
    root.title("Procesador Virtual IA")

    start_button = tk.Button(root, text="Iniciar", command=start)
    start_button.pack()

    stop_button = tk.Button(root, text="Detener", command=stop)
    stop_button.pack()

    root.mainloop()

if __name__ == "__main__":
    model = train_model()
    create_interface(model)
