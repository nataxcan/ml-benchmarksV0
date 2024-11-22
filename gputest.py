import torch

def list_cuda_devices():
    if not torch.cuda.is_available():
        print("No CUDA devices found.")
        return

    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices: {num_devices}")

    for i in range(num_devices):
        device_name = torch.cuda.get_device_name(i)
        print(f"Device {i}: {device_name}")

if __name__ == "__main__":
    list_cuda_devices()