import tensorflow as tf
import time

def gpu_benchmark(matrix_size=4096, iterations=10):
    """Benchmark GPU performance using matrix multiplication."""
    if not tf.config.list_physical_devices('GPU'):
        print("No GPU found! Ensure your GPU is properly set up with drivers and TensorFlow.")
        return

    print("Starting GPU Benchmark...")
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    print(f"Iterations: {iterations}\n")

    # Set up matrix multiplication
    with tf.device('/GPU:0'):
        A = tf.random.uniform((matrix_size, matrix_size), minval=0, maxval=1, dtype=tf.float32)
        B = tf.random.uniform((matrix_size, matrix_size), minval=0, maxval=1, dtype=tf.float32)

        # Warm-up to ensure everything is loaded
        print("Warming up GPU...")
        _ = tf.matmul(A, B)

        # Benchmark
        times = []
        for i in range(iterations):
            start_time = time.time()
            _ = tf.matmul(A, B)
            end_time = time.time()
            elapsed_time = end_time - start_time
            times.append(elapsed_time)
            print(f"Iteration {i + 1}: {elapsed_time:.4f} seconds")

    avg_time = sum(times) / len(times)
    print(f"\nAverage Time per Iteration: {avg_time:.4f} seconds")
    print(f"Performance: {matrix_size**3 / avg_time / 1e9:.2f} GFLOPS (approx.)")

if __name__ == "__main__":
    # Adjust matrix size and iterations as needed
    gpu_benchmark(matrix_size=4096, iterations=10)
