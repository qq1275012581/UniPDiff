import os
import numpy as np
import onnx
import onnxruntime as ort
import time
import pandas as pd

# 1. Enhanced Session Options
options = ort.SessionOptions()

# ENABLE these to reuse memory buffers and speed up inference
options.enable_cpu_mem_arena = True
options.enable_mem_pattern = True
# Set optimization level to the highest
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
# Set threads based on CPU cores for better scheduling
options.intra_op_num_threads = os.cpu_count()

# 2. Optimized CUDA Provider Options
cuda_provider_options = {
    'device_id': 0,
    'arena_extend_strategy': 'kSameAsRequested',
    'cudnn_conv_algo_search': 'DEFAULT', # Can use 'EXHAUSTIVE' for best speed but slow first step
    'do_copy_in_default_stream': True,
}

print("--- Initializing ONNX Runtime Session ---")
try:
    # Set CUDA as primary provider, CPU as fallback
    providers = [('CUDAExecutionProvider', cuda_provider_options), 'CPUExecutionProvider']
    ort_session_6 = ort.InferenceSession('pangu_weather_6.onnx', sess_options=options, providers=providers)
    print(f"Active Execution Providers: {ort_session_6.get_providers()}")
except Exception as e:
    print(f"Error during session initialization: {e}")

# Load input data
input_data_dir = '/home-ssd/Users/gm_intern/liguowen/UniPDiff/input_data'
output_data_dir = '/home-ssd/Users/gm_intern/liguowen/UniPDiff/output_data'
steps = 7 * 4

def pengu_inference(time_str):
    
    surf_output_file = os.path.join(output_data_dir, f"pangu_surface", f'{time_str}.npy')
    upper_output_file = os.path.join(output_data_dir, f"pangu_upper", f'{time_str}.npy')
    if os.path.exists(surf_output_file) and os.path.exists(upper_output_file):
        return None, None
    input_upper = np.load(os.path.join(input_data_dir, 'upper', f'{time_str}.npy')).astype(np.float32)
    input_surface = np.load(os.path.join(input_data_dir, 'surface', f'{time_str}.npy')).astype(np.float32)
    
    output, output_surface = input_upper, input_surface

    print(f"--- Starting Inference {time_str} for {steps} steps ---")
    all_outputs, all_outputs_surface = [], []
    start_time = time.time()
    for step in range(steps):
        # Model inference
        output, output_surface = ort_session_6.run(None, {'input': output, 'input_surface': output_surface})
        all_outputs.append(output)
        all_outputs_surface.append(output_surface)

    output = np.stack(all_outputs, axis=0)
    output_surface = np.stack(all_outputs_surface, axis=0)
    print(f"{time_str} Output shape after {steps} steps: {output.shape}, Surface shape: {output_surface.shape}")
    # Saving results
    
    if not os.path.exists(surf_output_file):
        os.makedirs(os.path.dirname(surf_output_file), exist_ok=True)
        np.save(surf_output_file, output_surface)
        pass

    
    if not os.path.exists(upper_output_file):
        os.makedirs(os.path.dirname(upper_output_file), exist_ok=True)
        np.save(upper_output_file, output)
        pass

    elapsed_time = time.time() - start_time
    return output_surface, output