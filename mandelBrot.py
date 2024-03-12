from numba import vectorize, complex64, boolean
import numpy as np


class mandelBrot():
    def __init__(self, max_iter, width, height):
        self.max_iter = max_iter
        self.width = width
        self.height = height
        
    @vectorize([boolean(complex64)], nopython=True)
    def compute(self, c):
        z = 0.0j
        for i in range(self.max_iter):
            if abs(z) > 2.0:
                return False
            z = z*z + c
        return True
            
        

    def map(self, x, y, scale_x, scale_y, delta_x, delta_y):
        real = -2 + ((x / self.width) * scale_x) + delta_x
        imag = -2 + ((y / self.height) * scale_y) + delta_y
        return complex(real, imag)