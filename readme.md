Here's a Python script to benchmark server-grade GPUs using TensorFlow. It performs a matrix multiplication task to stress the GPU, a common benchmarking method for machine learning workloads.



# Prerequisites

1. Install TensorFlow with GPU support:

pip install tensorflow


2. Ensure NVIDIA CUDA and cuDNN or AMD ROCm are properly installed.



# How It Works

1. Matrix Multiplication:

The script generates two large random matrices (matrix_size x matrix_size).

It performs matrix multiplication on the GPU to stress test it.



2. Benchmarking:

Measures the time taken for each multiplication.

Calculates approximate GFLOPS (Giga Floating-Point Operations Per Second).



3. Results:

Outputs the time for each iteration.

Reports the average time and performance in GFLOPS.



# Usage

1. Run the script:

python gpu_benchmark.py


2. Adjust matrix_size and iterations for more intensive or shorter tests:

Larger matrix sizes increase stress on the GPU.

More iterations give more precise averages.
