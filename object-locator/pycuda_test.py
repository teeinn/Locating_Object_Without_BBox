import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy
import cv2
import time
start = time.time()
test_array = numpy.random.randn(10000, 10000).astype(numpy.float32)
minn, maxx = test_array.min(), test_array.max()
array_scaled = ((test_array - minn)/(maxx - minn)*255) \
            .round().astype(numpy.uint8).squeeze()
tau, mask = cv2.threshold(array_scaled,
                          0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
tau = minn + (tau/255)*(maxx - minn)
end = time.time()
print("cpu: {}".format(end-start))


n = 10000 * 10000
blocksize = 1024
nblocks = numpy.float32(int(n / blocksize))

start = time.time()
test_array = numpy.random.randn(10000, 10000).astype(numpy.float32)
minn, maxx = test_array.min(), test_array.max()

test_array_gpu = cuda.mem_alloc(test_array.nbytes)
cuda.memcpy_htod(test_array_gpu, test_array)
#
# mod = SourceModule(""" __global__ void squeeze(float *a, int min, int max)
#                             { int idx = threadIdx.x + threadIdx.y*640;
#                               a[idx] = (a[idx]-min) / (max-min) * 255;
#                             } """)

mod = SourceModule(""" __global__ void squeeze(float *a, int min, int max, int n)
                            { int idx = threadIdx.x + blockIdx.x * blockDim.x;
                              if (idx < n)                                         
                                 a[idx] = (a[idx]-min) / (max-min) * 255;
                            } """)

func = mod.get_function("squeeze")
func(test_array_gpu, minn, maxx, nblocks, block=(16, 16, 1))

test_array_doubled = numpy.empty_like(test_array)
cuda.memcpy_dtoh(test_array_doubled, test_array_gpu)

final = test_array_doubled.round().astype(numpy.uint8).squeeze()
tau, mask = cv2.threshold(final, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
tau = minn + (tau/255)*(maxx - minn)
end = time.time()
print("gpu: {}".format(end-start))


