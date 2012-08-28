
'''
Based on code from:
http://enja.org/2011/02/22/adventures-in-pyopencl-part-1-getting-started-with-python/index.html
'''

import pyopencl as cl
import time

kernel = "__kernel void fcn(__global float* a, __global float* b, __global float* c)\
		{\
		    unsigned int i = get_global_id(0);\
		    c[i] = a[i] + b[i];\
		}"


ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)
program = cl.Program(ctx, kernel).build()
mf = cl.mem_flags

# Init cpu
# a = numpy.array(range(10000000), dtype=numpy.float32)
# b = numpy.array(range(10000000), dtype=numpy.float32)
a = np.ones([50,100], dtype=numpy.float32)
b = np.ones([50,100], dtype=numpy.float32)
c = np.empty_like(a) # For output

# Create OpenCL buffers
a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
dest_buf = cl.Buffer(ctx, mf.WRITE_ONLY, b.nbytes)

#Execute
c = np.empty_like(a) # For output
t1 = time.time()
program.fcn(queue, a.shape, None, a_buf, b_buf, dest_buf)
cl.enqueue_read_buffer(queue, dest_buf, c).wait()
print time.time() - t1

c = np.empty_like(a) # For output
t1 = time.time()
c2 = a+b
print time.time() - t1