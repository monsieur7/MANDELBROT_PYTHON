import pygame
import numpy as np
import pyopencl as cl
from pygame import _sdl2 as sdl2
import math
from matplotlib import colors
from functools import cache
with open("mandelbrot.cl", "r") as f:
    kernel_code = f.read()

def update_zoom(event, scale_x, scale_y, delta_x, delta_y, max_iter):
    if event.key == pygame.K_UP:
        scale_x *= 0.9
        scale_y *= 0.9
    elif event.key == pygame.K_DOWN:
        scale_x *= 1.1
        scale_y *= 1.1
    elif event.key == pygame.K_z:
        delta_y -= 0.1 / scale_y
    elif event.key == pygame.K_s:
        delta_y += 0.1/scale_y
    elif event.key == pygame.K_q:
        delta_x -= 0.1/scale_x
    elif event.key == pygame.K_d:
        delta_x += 0.1/scale_x
    elif event.key == pygame.K_r:
        delta_x = 0
        delta_y = 0
        max_iter = 100
    elif event.key == pygame.K_i:
        max_iter += 10
    elif event.key == pygame.K_o:
        max_iter -= 10
    return scale_x, scale_y, delta_x, delta_y, max_iter
def largest_divisor_less_than(number, limit):
    largest_divisor = 1
    for divisor in range(1, limit+1):
        if number % divisor == 0:
            largest_divisor = divisor
    return largest_divisor
width = 800
height = 800
scale_x = 1
scale_y = 1
delta_x = 0
delta_y = 0
max_iter = 100
init:bool = False
initial_aspect_ratio = width / height

x = np.linspace(-2/scale_x + delta_x, 2/scale_x + delta_x, width, dtype=np.float32)
y = np.linspace(-2/scale_y + delta_y, 2/scale_y + delta_y, height, dtype=np.float32)

clock = pygame.time.Clock()

screen_array = np.zeros((width, height, 3), dtype=np.uint8)
output = np.zeros((width, height), dtype=np.float32)

if (__name__ == "__main__"):
    
    platform = cl.get_platforms()[0]
    device = platform.get_devices()[0]
    
    print(f"device with max work group size {device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)} and max work item sizes {device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)}")
    print(f"device with {device.get_info(cl.device_info.MAX_COMPUTE_UNITS)} compute units ")
    max_work_group_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
    max_work_item_sizes = device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)

    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    
    
    print(f"Using device {device.name} on platform {platform.name} device type: {device.type}")
    program = cl.Program(context, kernel_code).build()

    pygame.init()
    pygame.display.set_caption("Mandelbrot Set")
    
    window = pygame.display.set_mode((width, height), pygame.RESIZABLE | pygame.DOUBLEBUF)
    running = True
    init = False
    sdl_window = sdl2.Window.from_display_module()
    render = sdl2.Renderer.from_window(sdl_window)
    font = pygame.font.Font(None, 36)

    while(running):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                scale_x, scale_y, delta_x, delta_y, max_iter = update_zoom(event, scale_x, scale_y, delta_x, delta_y, max_iter)
                x = np.linspace(-2/scale_x + delta_x, 2/scale_x + delta_x, width, dtype=np.float32) # update x and y arrays   
                y = np.linspace(-2/scale_y + delta_y, 2/scale_y + delta_y, height, dtype=np.float32)
                init = False
            if event.type == pygame.VIDEORESIZE:
                width, height = event.w, event.h

                # Update the aspect ratio
                new_aspect_ratio = width / height

                # Update the scaling factors to maintain the aspect ratio
                if new_aspect_ratio > initial_aspect_ratio:
                    scale_y = scale_x * new_aspect_ratio
                elif new_aspect_ratio < initial_aspect_ratio:
                    scale_x = scale_y / new_aspect_ratio
                

                # Update the range of coordinates used for calculating the Mandelbrot set
                x_min = -2 / scale_x + delta_x
                x_max = 2 / scale_x + delta_x
                y_min = -2 / scale_y + delta_y
                y_max = 2 / scale_y + delta_y

                x = np.linspace(x_min, x_max, width, dtype=np.float32)
                y = np.linspace(y_min, y_max, height, dtype=np.float32)
                # Update the screen array and output size
                screen_array = np.zeros((width, height, 3), dtype=np.uint8)
                output = np.zeros((width, height), dtype=np.float32)
                init = False

               


                print(f"New width: {width}, New height: {height}, New aspect ratio: {new_aspect_ratio:.2f}")

                
        
        if not init:
            # Create the OpenCL buffers
            output_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, output.nbytes)
            x_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=x)
            y_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=y)
            item_size = (largest_divisor_less_than(width, int(math.sqrt(max_work_group_size))), largest_divisor_less_than(height, int(math.sqrt(max_work_group_size))))
            work_size = (width, height)
            print(f"item size {item_size} work size {work_size}")
            init = True

        program.mandelbrot(queue, work_size, item_size, output_buf, x_buf, y_buf, np.int32(width), np.int32(height), np.int32(max_iter))
        
        cl.enqueue_copy(queue, output, output_buf).wait()
        
       
        screen_array[:,:,0] = output * 1
        screen_array[:,:,1] = output * 2
        screen_array[:,:,2] = output * 5
        
       

        
        #create a surface :
        surface = pygame.surfarray.make_surface(screen_array)
        texture = sdl2.Texture.from_surface(render, surface)
        render.clear()
        texture.draw(dstrect=window.get_rect())   
        #draw into the window : 
        
        
        text = font.render(f"{clock.get_fps():.2f}fps Iters {max_iter}", True, (0, 255, 0))
        text_texture = sdl2.Texture.from_surface(render, text)
        
        text_texture.draw(srcrect=text_texture.get_rect(), dstrect=text_texture.get_rect())
        render.present() 

        clock.tick()
        
       
        
output_buf.release()
x_buf.release()
y_buf.release()
        
pygame.quit()
        
