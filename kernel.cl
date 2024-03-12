__kernel void add_arrays(__global const float *a, __global const float *b,
                         __global float *result, const int size) {
  int i = get_global_id(0);
  if (i < size)
    result[i] = a[i] + b[i];
}