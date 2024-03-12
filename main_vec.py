from numba import vectorize, boolean, float64, njit, jit, prange
import numba
#import numba_dpex as dpex
import numpy as np
import pygame
#taichi : to see !
clock = pygame.time.Clock()

@njit(fastmath=True, parallel=True, cache=True)
#@dpex.dpjit()   
def compute_mandelbrot(x, y, max_iter, screen_array):
    print("Using dpex")
    print(f"X shape: {x.shape[0]} Y shape: {y.shape[0]}")
    for i in prange(x.shape[0]):
        for j in prange(y.shape[0]):
            c = complex(x[i], y[j])
            z = 0.0j
            for k in range(max_iter):
                if z.real*z.real + z.imag*z.imag >= 4:
                    #add some nice colors
                    screen_array[i, j] = [k*2, k*3, k*5]
                    break
                z = z*z + c
            
    return screen_array
@njit(fastmath=True, parallel=True, cache=True)
def compute_julia(x, y, max_iter, screen_array):
    for i in numba.prange(x.shape[0]):
        for j in numba.prange(y.shape[0]):
            c = complex(-0.038088, 0.9754633)
            z = complex(x[i], y[j])
            for k in range(max_iter):
                if z.real*z.real + z.imag*z.imag >= 4:
                    #add some nice colors
                    screen_array[i, j] = [k*2, k*3, k*5]
                    break
                z = z*z + c
            #add some nice colors (others)
            
    return screen_array

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
        scale_x = 4
        scale_y = 4
        delta_x = 0
        delta_y = 0
        max_iter = 100
    elif event.key == pygame.K_i:
        max_iter += 10
    elif event.key == pygame.K_o:
        max_iter -= 10
    return scale_x, scale_y, delta_x, delta_y, max_iter

width = 800
height = 800
scale_x = 1
scale_y = 1
delta_x = 0
delta_y = 0
max_iter = 100

x = np.linspace(-2/scale_x + delta_x, 2/scale_x + delta_x, width)
y = np.linspace(-2/scale_y + delta_y, 2/scale_y + delta_y, height)

screen_array = np.zeros((width, height, 3), dtype=np.uint8)

pygame.init()
pygame.display.set_caption("Mandelbrot Set")
screen = pygame.display.set_mode((width, height), pygame.RESIZABLE | pygame.DOUBLEBUF)
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            scale_x, scale_y, delta_x, delta_y, max_iter = update_zoom(event, scale_x, scale_y, delta_x, delta_y, max_iter)
            x = np.linspace(-2/scale_x + delta_x, 2/scale_x + delta_x, width) # update x and y arrays   
            y = np.linspace(-2/scale_y + delta_y, 2/scale_y + delta_y, height)
        if event.type == pygame.VIDEORESIZE:
            width = event.w
            height = event.h 
            if(width / height > 1):
                #adjust scale_y to keep the same ratio
                scale_y = scale_x * (width / height)
            elif(width / height < 1):
                #adjust scale_x to keep the same ratio
                scale_x = scale_y * (height / width)
            
            screen_array = np.zeros((width, height, 3), dtype=np.uint8)
            x = np.linspace(-2/scale_x + delta_x, 2/scale_x + delta_x, width)
            y = np.linspace(-2/scale_y + delta_y, 2/scale_y + delta_y, height)
            
            print(f" New width: {width} New height: {height}")
        
    screen_array.fill(0)
    #compute_mandelbrot.parallel_diagnostics(level=4)
    result = compute_mandelbrot(x, y, max_iter, screen_array)
    #result = compute_julia(x, y, max_iter, screen_array)
    
    surf = pygame.surfarray.make_surface(result)
    screen.blit(surf, (0, 0))
    
    #fps counter at top left
    
    font = pygame.font.Font(None, 36)
    text = font.render(f"{clock.get_fps():.2f}fps Iters {max_iter}", True, (255, 0, 0))
    screen.blit(text, (0, 0))

    pygame.display.flip()
    clock.tick()

pygame.quit()
