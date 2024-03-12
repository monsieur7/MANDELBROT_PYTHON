import pyopencl as cl
import numpy as np

# Define the kernel code for adding two arrays element-wise
with open("kernel.cl", "r") as f:
    kernel_code = f.read()
    
def main():
    # Initialize OpenCL context and command queue
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    print(f"Using device {device.name} on platform {platform.name} device type: {device.type}")

    # Define input arrays
    a = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    b = np.array([6, 7, 8, 9, 10], dtype=np.float32)
    size = a.shape[0]

    # Create OpenCL buffers
    a_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=b)
    result_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, b.nbytes)

    # Compile the kernel program
    program = cl.Program(context, kernel_code).build()

    # Execute the kernel
    program.add_arrays(queue, a.shape, None, a_buf, b_buf, result_buf, np.int32(size))
    
    # Read the result back from the GPU
    result = np.empty_like(a)
    cl.enqueue_copy(queue, result, result_buf).wait()

    print("Array A:", a)
    print("Array B:", b)
    print("Result:", result)

if __name__ == "__main__":
    main()
