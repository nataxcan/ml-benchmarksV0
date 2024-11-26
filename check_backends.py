import torch

# Define a simple model to test
class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

# Instantiate the model
model = TestModel()
x = torch.randn(5, 10)  # Example input

# List all available backends
available_backends = torch._dynamo.list_backends()
print("Testing available backends for torch.compile:")
print(available_backends)

# Test each backend
for backend in available_backends:
    if backend != 'openxla':
        print("="*100)
        print(f"\nTesting backend: {backend}")
        try:
            # Compile the model with the current backend
            compiled_model = torch.compile(model, backend=backend)
            # Run the compiled model
            output = compiled_model(x)
            print(f"Backend '{backend}' works successfully.")
        except Exception as e:
            print(f"Backend '{backend}' failed with error: {e}")