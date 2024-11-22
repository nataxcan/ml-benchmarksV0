import torch
import torch.nn as nn
import torch.jit
import time
import gc
import sys
import traceback
import pandas as pd

# Import your models here
from models.grus import GRU, JitGRU
from models.lstms import (
    LSTM, LSTMFaster, LSTMUnrolled, JitLSTM,
    OptimizedCustomLSTM, OptimizedCustomLSTM2, OptimizedCustomLSTM3
)
from models.parallelgrus import (
    ParallelGRULayer, ParallelGRULayerv1, ParallelGRULayerv2,
    ParallelGRULayerv3, ParallelGRULayerv4, ParallelGRULayerv5,
    ParallelGRULayerv6, ParallelGRULayerv7, ParallelGRULayerv8,
    ParallelGRULayerv9
)
from models.abl_grus import get_jitgru_ablations

# Ensure that CUDA is available
assert torch.cuda.is_available(), "CUDA is not available. Please check your CUDA installation."

SEQ_LEN = 500
INPUT_SIZE = 384
HIDDEN_SIZE = 384

num_filters = INPUT_SIZE
num_recurrent_steps = 3

# Define functions to instantiate models
def get_model_GRU():
    return GRU(INPUT_SIZE, HIDDEN_SIZE)

def get_model_JitGRU():
    return JitGRU(INPUT_SIZE, HIDDEN_SIZE, num_layers=1, batch_first=True)

def get_model_LSTM():
    return LSTM(INPUT_SIZE, HIDDEN_SIZE)

def get_model_LSTMFaster():
    return LSTMFaster(INPUT_SIZE, HIDDEN_SIZE)

def get_model_LSTMUnrolled():
    return LSTMUnrolled(INPUT_SIZE, HIDDEN_SIZE, sequence_length=SEQ_LEN)

def get_model_JitLSTM():
    return JitLSTM(INPUT_SIZE, HIDDEN_SIZE, num_layers=1, batch_first=True)

def get_model_OptimizedCustomLSTM():
    return OptimizedCustomLSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)

def get_model_OptimizedCustomLSTM2():
    return OptimizedCustomLSTM2(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)

def get_model_OptimizedCustomLSTM3():
    return OptimizedCustomLSTM3(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)

def get_model_ParallelGRULayer():
    return ParallelGRULayer(num_filters, num_recurrent_steps)

def get_model_ParallelGRULayerv1():
    return ParallelGRULayerv1(num_filters, num_recurrent_steps)

def get_model_ParallelGRULayerv2():
    return ParallelGRULayerv2(num_filters, num_recurrent_steps)

def get_model_ParallelGRULayerv3():
    return ParallelGRULayerv3(num_filters, num_recurrent_steps)

def get_model_ParallelGRULayerv4():
    return ParallelGRULayerv4(num_filters, num_recurrent_steps)

def get_model_ParallelGRULayerv5():
    return ParallelGRULayerv5(num_filters, num_recurrent_steps)

def get_model_ParallelGRULayerv6():
    return ParallelGRULayerv6(num_filters, num_recurrent_steps)

def get_model_ParallelGRULayerv7():
    return ParallelGRULayerv7(num_filters, num_recurrent_steps)

def get_model_ParallelGRULayerv8():
    return ParallelGRULayerv8(num_filters, num_recurrent_steps)

def get_model_ParallelGRULayerv9():
    return ParallelGRULayerv9(num_filters, num_recurrent_steps)

def get_model_JitGRUAblations_noz():
    return get_jitgru_ablations('noz', INPUT_SIZE, HIDDEN_SIZE)

def get_model_JitGRUAblations_nor():
    return get_jitgru_ablations('nor', INPUT_SIZE, HIDDEN_SIZE)

def get_model_JitGRUAblations_nozr():
    return get_jitgru_ablations('nozr', INPUT_SIZE, HIDDEN_SIZE)

def get_model_JitGRUAblations_onezr():
    return get_jitgru_ablations('onezr', INPUT_SIZE, HIDDEN_SIZE)

# List of models to benchmark
models_to_benchmark = [
    {'name': 'GRU', 'get_model': get_model_GRU},
    {'name': 'JitGRU', 'get_model': get_model_JitGRU},
    {'name': 'LSTM', 'get_model': get_model_LSTM},
    {'name': 'LSTMFaster', 'get_model': get_model_LSTMFaster},
    {'name': 'LSTMUnrolled', 'get_model': get_model_LSTMUnrolled},
    {'name': 'JitLSTM', 'get_model': get_model_JitLSTM},
    {'name': 'OptimizedCustomLSTM', 'get_model': get_model_OptimizedCustomLSTM},
    {'name': 'OptimizedCustomLSTM2', 'get_model': get_model_OptimizedCustomLSTM2},
    {'name': 'OptimizedCustomLSTM3', 'get_model': get_model_OptimizedCustomLSTM3},
    {'name': 'ParallelGRULayer', 'get_model': get_model_ParallelGRULayer},
    {'name': 'ParallelGRULayerv1', 'get_model': get_model_ParallelGRULayerv1},
    {'name': 'ParallelGRULayerv2', 'get_model': get_model_ParallelGRULayerv2},
    {'name': 'ParallelGRULayerv3', 'get_model': get_model_ParallelGRULayerv3},
    {'name': 'ParallelGRULayerv4', 'get_model': get_model_ParallelGRULayerv4},
    {'name': 'ParallelGRULayerv5', 'get_model': get_model_ParallelGRULayerv5},
    {'name': 'ParallelGRULayerv6', 'get_model': get_model_ParallelGRULayerv6},
    {'name': 'ParallelGRULayerv7', 'get_model': get_model_ParallelGRULayerv7},
    {'name': 'ParallelGRULayerv8', 'get_model': get_model_ParallelGRULayerv8},
    {'name': 'ParallelGRULayerv9', 'get_model': get_model_ParallelGRULayerv9},
    {'name': 'JitGRUAblations_noz', 'get_model': get_model_JitGRUAblations_noz},
    {'name': 'JitGRUAblations_nor', 'get_model': get_model_JitGRUAblations_nor},
    {'name': 'JitGRUAblations_nozr', 'get_model': get_model_JitGRUAblations_nozr},
    {'name': 'JitGRUAblations_onezr', 'get_model': get_model_JitGRUAblations_onezr},
]

# Settings to test
settings_list = ['default', 'jit_script', 'compile_default', 'compile_cudagraphs']

results = []

def benchmark_model(model, model_name, batch_size, setting):
    # Flush GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    try:
        model = model.to('cuda')

        if setting == 'default':
            pass  # Use the model as is
        elif setting == 'jit_script':
            model = torch.jit.script(model)
        elif setting == 'compile_default':
            model = torch.compile(model)
        elif setting == 'compile_cudagraphs':
            model = torch.compile(model, backend='cudagraphs')
        else:
            print(f"Unknown setting {setting}")
            return None

        # Prepare inputs
        input_shape = (batch_size, SEQ_LEN, INPUT_SIZE)
        input_dtype = torch.float32
        inputs = [torch.randn(input_shape, dtype=input_dtype, device='cuda') for _ in range(2)]

        # Warm-up iterations
        num_warmup = 10
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(inputs[0])

        torch.cuda.synchronize()

        # Benchmark
        num_repetitions = 50
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        with torch.no_grad():
            for i in range(num_repetitions):
                _ = model(inputs[i % 2])

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_time_s = elapsed_time_ms / 1000.0

        # Calculate throughput
        total_input_vectors = batch_size * num_repetitions * SEQ_LEN
        throughput = total_input_vectors / elapsed_time_s

        # Get model description
        model_str = str(model)

        # Flush GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        return {
            'model_name': model_name,
            'setting': setting,
            'batch_size': batch_size,
            'throughput': throughput,
            'model_str': model_str
        }

    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"OOM for model {model_name}, setting {setting}, batch size {batch_size}")
            torch.cuda.empty_cache()
            gc.collect()
            return None
        else:
            print(f"RuntimeError for model {model_name}, setting {setting}, batch size {batch_size}: {e}")
            traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()
            return None
    except Exception as e:
        print(f"Exception during benchmarking model {model_name}, setting {setting}, batch size {batch_size}: {e}")
        traceback.print_exc()
        torch.cuda.empty_cache()
        gc.collect()
        return None

# Main benchmarking loop
for model_info in models_to_benchmark:
    model_name = model_info['name']
    get_model = model_info['get_model']
    print(f"\nBenchmarking model {model_name}")
    for setting in settings_list:
        batch_size = 2
        while True:
            model = get_model()
            result = benchmark_model(model, model_name, batch_size, setting)
            if result is not None:
                results.append(result)
                print(f"Model {model_name}, setting {setting}, batch size {batch_size}: Throughput {result['throughput']:.2f} vectors/sec")
                # Increase batch size
                if batch_size < 1024:
                    batch_size *= 2
                else:
                    batch_size += 256
            else:
                print(f"Model {model_name}, setting {setting}, batch size {batch_size} failed or OOM.")
                break  # Exit the loop when OOM occurs

# Save results to a CSV file
df_results = pd.DataFrame(results)
df_results.to_csv('benchmark_results.csv', index=False)
print("\nBenchmarking completed. Results saved to 'benchmark_results.csv'")
