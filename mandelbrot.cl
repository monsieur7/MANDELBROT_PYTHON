__kernel void mandelbrot(__global float *output, __global float* x, __global float* y, const int width, const int height, const int iterations){
    float2 z = (float2)(0.0f, 0.0f);
    int x_index = get_global_id(0);
    int y_index = get_global_id(1);

    int index = x_index * height + y_index;
    float2 c = (float2)(x[x_index], y[y_index]);

    if(x_index >= width || y_index >= height){
        return;
    }
    if(x_index < 0 || y_index < 0){
        return;
    }

    int i;
    for(i = 0; i < iterations; i++){
        z = (float2)(z.x * z.x - z.y * z.y, 2.0f * z.x * z.y) + c;
        if(z.x * z.x + z.y * z.y > 4.0f){
            break;
        }
    }
    
   if(i < iterations){

        float v = i + 1 - log(log(sqrt(z.x * z.x + z.y * z.y))) / log((float)(2));
        output[index] = v;
    }else{
        output[index] =(float) i;
    }



}
