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

try:
    torch.set_float32_matmul_precision('high')
except Exception as e:
    print("could not set float32 tensor core stuff because", e)

SEQ_LEN = 500
INPUT_SIZE = 384
HIDDEN_SIZE = 384
SAVE_LOCATION = 'train_benchmark_results_batchfirst0.csv'

TRAIN = True

num_filters = INPUT_SIZE
num_recurrent_steps = 3
BATCH_FIRST = False

# Define functions to instantiate models
def get_model_TorchGRU():
    return nn.GRU(INPUT_SIZE, HIDDEN_SIZE, batch_first=BATCH_FIRST)

def get_model_TorchLSTM():
    return nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=BATCH_FIRST)

def get_model_GRU():
    return GRU(INPUT_SIZE, HIDDEN_SIZE)

def get_model_JitGRU():
    return JitGRU(INPUT_SIZE, HIDDEN_SIZE, num_layers=1, batch_first=BATCH_FIRST)

def get_model_LSTM():
    return LSTM(INPUT_SIZE, HIDDEN_SIZE)

def get_model_LSTMFaster():
    return LSTMFaster(INPUT_SIZE, HIDDEN_SIZE)

def get_model_LSTMUnrolled():
    return LSTMUnrolled(INPUT_SIZE, HIDDEN_SIZE, sequence_length=SEQ_LEN)

def get_model_JitLSTM():
    return JitLSTM(INPUT_SIZE, HIDDEN_SIZE, num_layers=1, batch_first=BATCH_FIRST)

def get_model_OptimizedCustomLSTM():
    return OptimizedCustomLSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=BATCH_FIRST)

def get_model_OptimizedCustomLSTM2():
    return OptimizedCustomLSTM2(INPUT_SIZE, HIDDEN_SIZE, batch_first=BATCH_FIRST)

def get_model_OptimizedCustomLSTM3():
    return OptimizedCustomLSTM3(INPUT_SIZE, HIDDEN_SIZE, batch_first=BATCH_FIRST)

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
    {'name': 'TorchGRU', 'get_model': get_model_TorchGRU},
    {'name': 'TorchLSTM', 'get_model': get_model_TorchLSTM},
    {'name': 'GRU', 'get_model': get_model_GRU},
    {'name': 'JitGRU', 'get_model': get_model_JitGRU},
    {'name': 'LSTM', 'get_model': get_model_LSTM},
    {'name': 'LSTMFaster', 'get_model': get_model_LSTMFaster},
    {'name': 'LSTMUnrolled', 'get_model': get_model_LSTMUnrolled},
    {'name': 'JitLSTM', 'get_model': get_model_JitLSTM},
    {'name': 'OptimizedCustomLSTM', 'get_model': get_model_OptimizedCustomLSTM},
    {'name': 'OptimizedCustomLSTM2', 'get_model': get_model_OptimizedCustomLSTM2},
    {'name': 'OptimizedCustomLSTM3', 'get_model': get_model_OptimizedCustomLSTM3},
    {'name': 'JitGRUAblations_noz', 'get_model': get_model_JitGRUAblations_noz},
    {'name': 'JitGRUAblations_nor', 'get_model': get_model_JitGRUAblations_nor},
    {'name': 'JitGRUAblations_nozr', 'get_model': get_model_JitGRUAblations_nozr},
    {'name': 'JitGRUAblations_onezr', 'get_model': get_model_JitGRUAblations_onezr},

    # {'name': 'ParallelGRULayer', 'get_model': get_model_ParallelGRULayer},
    # {'name': 'ParallelGRULayerv1', 'get_model': get_model_ParallelGRULayerv1},
    # {'name': 'ParallelGRULayerv2', 'get_model': get_model_ParallelGRULayerv2},
    # {'name': 'ParallelGRULayerv3', 'get_model': get_model_ParallelGRULayerv3},
    # {'name': 'ParallelGRULayerv4', 'get_model': get_model_ParallelGRULayerv4},
    # {'name': 'ParallelGRULayerv5', 'get_model': get_model_ParallelGRULayerv5},
    # {'name': 'ParallelGRULayerv6', 'get_model': get_model_ParallelGRULayerv6},
    # {'name': 'ParallelGRULayerv7', 'get_model': get_model_ParallelGRULayerv7},
    # {'name': 'ParallelGRULayerv8', 'get_model': get_model_ParallelGRULayerv8},
    # {'name': 'ParallelGRULayerv9', 'get_model': get_model_ParallelGRULayerv9},
]

# Settings to test
settings_list = [
    'default',
    'jit_script',
    # 'compile_default',
    # 'compile_cudagraphs'
]

# Load existing results
try:
    existing_results = pd.read_csv(SAVE_LOCATION)
    # Create a set of existing combinations
    existing_combinations = set(zip(existing_results['model_name'], existing_results['setting'], existing_results['batch_size']))
except FileNotFoundError:
    existing_results = pd.DataFrame()
    existing_combinations = set()
except pd.errors.EmptyDataError:
    existing_results = pd.DataFrame()
    existing_combinations = set()

results = []

def tuple_to_tensor(t):
    if isinstance(t, tuple):
        return tuple_to_tensor(t[0])
    else:
        return t

def benchmark_model(model, model_name, batch_size, setting, last_tot_time, optimizer):
    # Flush GPU memory
    loss_fn = nn.MSELoss()
    torch.cuda.empty_cache()
    gc.collect()
    if last_tot_time > 20:
        print('last run was too long at', last_tot_time, 'seconds')
        return None
    num_repetitions = 10 if last_tot_time < 5 else 5
    num_warmup = 3 if last_tot_time < 5 else 2
    num_repetitions = num_repetitions if last_tot_time < 9 else 2
    num_warmup = num_warmup if last_tot_time < 9 else 1
    num_repetitions = num_repetitions if last_tot_time < 10 else 1
    num_warmup = num_warmup if last_tot_time < 10 else 0
    torch.cuda.reset_peak_memory_stats()

    try:

        # Prepare inputs
        input_shape = (batch_size, SEQ_LEN, INPUT_SIZE) # for RNNs
        # input_shape = (SEQ_LEN, batch_size, INPUT_SIZE) # for RNNs with batchfirst off
        # input_shape = (batch_size, INPUT_SIZE, SEQ_LEN) # for parallel archs
        input_dtype = torch.float32
        inputs = [torch.randn(input_shape, dtype=input_dtype, device='cuda') for _ in range(2)]

        target = None
        # Warm-up iterations
        if not TRAIN:
            with torch.no_grad():
                for _ in range(num_warmup):
                    out = model(inputs[1])
        else:
            for _ in range(num_warmup):
                if TRAIN:
                    optimizer.zero_grad()
                out = model(inputs[1])
                if TRAIN:
                    out = tuple_to_tensor(out)
                    target = torch.ones_like(out, device=out.device)
                    # print(target, out)
                    loss = loss_fn(out, target)
                    loss.backward()
                    optimizer.step()


        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        if TRAIN:
            for i in range(num_repetitions):
                if TRAIN:
                    optimizer.zero_grad()
                out = model(inputs[i % 2])
                if TRAIN:
                    out = tuple_to_tensor(out)
                    loss = loss_fn(out, target)
                    loss.backward()
                    optimizer.step()
        else:
            with torch.no_grad():
                for i in range(num_repetitions):
                    out = model(inputs[i % 2])

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_time_s = elapsed_time_ms / 1000.0
        max_memory = torch.cuda.max_memory_allocated()
        max_memory_gb = max_memory / (1024 ** 3)

        # Calculate throughput
        total_input_vectors = batch_size * num_repetitions * SEQ_LEN
        throughput = total_input_vectors / elapsed_time_s
        
        # so we can avoid shared memory usage
        if max_memory_gb > 20:
            elapsed_time_s = 100.0

        # Get model description
        model_str = str(model)

        # Flush GPU memory
        del inputs
        torch.cuda.empty_cache()
        gc.collect()

        return {
            'model_name': model_name,
            'setting': setting,
            'batch_size': batch_size,
            'throughput': throughput,
            'model_str': model_str,
            'tot_time': elapsed_time_s,
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
        last_tot_time = 0
        try:
            model = get_model()
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
        except Exception as e:
            print('could not compile model with setting', setting, 'because', e)
            continue
            
        optimizer = None
        if TRAIN:
            optimizer = torch.optim.Adam(model.parameters())
        
        while True:
            # Check if this combination has already been benchmarked
            if (model_name, setting, batch_size) in existing_combinations:
                print(f"Skipping Model {model_name}, setting {setting} as it's already benchmarked.")
                break

            result = benchmark_model(model, model_name, batch_size, setting, last_tot_time, optimizer)
            if result is not None:
                results.append(result)
                last_tot_time = max(result['tot_time'], last_tot_time)
                print(f"Model {model_name}, setting {setting}, batch size {batch_size}: Throughput {result['throughput']:.2f} vectors/sec, tot time: {result['tot_time']:.2f}")
                # Increase batch size
                if batch_size < 1024:
                    batch_size *= 2
                else:
                    batch_size += 512
            else:
                print(f"Model {model_name}, setting {setting}, batch size {batch_size} failed or OOM.")
                break  # Exit the loop when OOM occurs

# Combine existing results with new results
if not existing_results.empty:
    results_df = pd.DataFrame(results)
    combined_results = pd.concat([existing_results, results_df], ignore_index=True)
else:
    combined_results = pd.DataFrame(results)

# Save combined results to a CSV file
combined_results.to_csv(SAVE_LOCATION, index=False)
print("\nBenchmarking completed. Results saved to 'benchmark_results.csv'")