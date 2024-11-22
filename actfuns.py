import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

# Ensure GPU is available
assert torch.cuda.is_available(), "CUDA is not available. Please check your CUDA installation."

# List of activation functions to benchmark
activation_functions = {
    'ReLU': F.relu,
    'Sigmoid': torch.sigmoid,
    'Tanh': torch.tanh,
    'LeakyReLU': F.leaky_relu,
    'ELU': F.elu,
    'SELU': F.selu,
    'Softplus': F.softplus,
    'GELU': F.gelu,
    'Softsign': F.softsign,
    'Softmax': lambda x: F.softmax(x, dim=-1),
    'LogSoftmax': lambda x: F.log_softmax(x, dim=-1),
    'LogSigmoid': torch.nn.LogSigmoid(),
    'Hardtanh': F.hardtanh,
    'PReLU': nn.PReLU().to('cuda'),
    'RReLU': nn.RReLU(),
    'Tanhshrink': F.tanhshrink,
    'Softmin': lambda x: F.softmin(x, dim=-1),
    'ReLU6': F.relu6,
    'Softshrink': F.softshrink
}

# Input tensor
input_tensor = torch.randn(1000, 1000, device='cuda')

# Number of iterations
num_iterations = 10

# Dictionary to store the execution times
execution_times = {}

# Function to benchmark a single activation function
def benchmark_activation(func, input_tensor, num_iterations):
    with torch.no_grad():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        # Warm-up
        for _ in range(1000):
            _ = func(input_tensor)
        
        torch.cuda.synchronize()
        start_event.record()
        
        for _ in range(num_iterations):
            _ = func(input_tensor)
        
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
        return elapsed_time

# Benchmark each activation function
for name, func in activation_functions.items():
    time_taken = benchmark_activation(func, input_tensor, num_iterations)
    execution_times[name] = time_taken
    print(f"{name}: {time_taken:.2f} ms")

# Sorting the execution times
sorted_execution_times = dict(sorted(execution_times.items(), key=lambda item: item[1]))

# Plotting the results
plt.figure(figsize=(12, 6))
plt.bar(sorted_execution_times.keys(), sorted_execution_times.values(), color='skyblue')
plt.xlabel('Activation Functions')
plt.ylabel('Execution Time (ms)')
plt.title('Execution Time of Activation Functions (1000 iterations on GPU)')
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig("actfuns.png")