import numba_dpex as dpex
import dpnp
import dpctl
import matplotlib.pyplot as plt

@dpex.kernel
#simple add : c = a + b
def add(a, b, c):
    i = dpex.get_global_id(0)
    c[i] = a[i] + b[i]
gpu = dpctl.select_gpu_device()
gpu.print_device_info()
width = 800
height = 800
a = dpnp.arange(width*height, device=gpu)
b = dpnp.arange(width*height, device=gpu)
c = dpnp.zeros(width*height, device=gpu)

print(f"Using device {a.device}")
add[dpex.Range(width*height)](a, b, c)
print(c)