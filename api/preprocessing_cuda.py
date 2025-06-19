import base64
import numpy as np
import cv2
from io import BytesIO
import time
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import threading

# Thread-local storage for CUDA context
_thread_local = threading.local()

class CudaContextManager:
    """A context manager to handle CUDA context for each thread."""
    def __enter__(self):
        if not hasattr(_thread_local, 'context'):
            cuda.init()
            device = cuda.Device(0)
            _thread_local.context = device.make_context()
        else:
            _thread_local.context.push()
        return _thread_local.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(_thread_local, 'context'):
            _thread_local.context.pop()

# --- Helper Functions ---

def decode_image_base64(img_data: str):
    """Decodes a base64 string to a BGR numpy array."""
    byte_arr = base64.b64decode(img_data)
    img_np = np.frombuffer(byte_arr, np.uint8)
    return cv2.imdecode(img_np, cv2.IMREAD_COLOR)

def nparray_to_pngbase64str(img: np.ndarray):
    """Encodes a numpy array to a PNG base64 string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

# --- CUDA Kernel for Gamma Correction ---

def apply_lut_pycuda(img_data: str, lut_value: float) -> str:
    """Applies Gamma Correction using a custom PyCUDA kernel."""
    mod_apply_lut = SourceModule("""
    __global__ void apply_lut_kernel(unsigned char *dest, const unsigned char *src, const unsigned char *lut, int width, int height, int channels)
    {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = blockDim.x * gridDim.x;

            for (int i = idx; i < width * height * channels; i += stride)
            {
                dest[i] = lut[src[i]];
            }
        }
        """)
    img = decode_image_base64(img_data)
    if not isinstance(img, np.ndarray): return img_data
    img = img.astype(np.uint8)
    height, width, channels = img.shape

    lut_cpu = np.array([np.clip(pow(i / 255.0, lut_value) * 255.0, 0, 255) for i in range(256)], dtype=np.uint8)

    img_gpu = cuda.mem_alloc(img.nbytes)
    lut_gpu = cuda.mem_alloc(lut_cpu.nbytes)
    dest_gpu = cuda.mem_alloc(img.nbytes)

    cuda.memcpy_htod(img_gpu, img)
    cuda.memcpy_htod(lut_gpu, lut_cpu)

    apply_lut_kernel = mod_apply_lut.get_function("apply_lut_kernel")

    block_size = 256
    grid_size = (width * height * channels + block_size - 1) // block_size

    start_time = time.perf_counter()
    apply_lut_kernel(
        dest_gpu, img_gpu, lut_gpu,
        np.int32(width), np.int32(height), np.int32(channels),
        block=(block_size, 1, 1), grid=(grid_size, 1)
    )
    cuda.Context.synchronize()
    end_time = time.perf_counter()
    print(f"[PyCUDA] apply_lut - Time: {(end_time - start_time) * 1000:.3f} ms")

    result_img = np.empty_like(img)
    cuda.memcpy_dtoh(result_img, dest_gpu)

    img_gpu.free()
    lut_gpu.free()
    dest_gpu.free()

    return nparray_to_pngbase64str(result_img)

# --- CUDA Kernel for Image Rotation ---

def transformation_degrees_pycuda(img_data: str, degree_int: int) -> str:
    """The PyCUDA implementation of lossless 90/180/270 degree rotations."""
    mod_rotate = SourceModule("""
    __global__ void lossless_rotate_kernel(unsigned char* dest, const unsigned char* src, int src_w, int src_h, int channels, int mode)
    {
        int dest_x = blockIdx.x * blockDim.x + threadIdx.x;
        int dest_y = blockIdx.y * blockDim.y + threadIdx.y;

        int dest_w;
        if (mode == 90 || mode == 270) { dest_w = src_h; }
        else { dest_w = src_w; }
        int dest_h;
        if (mode == 90 || mode == 270) { dest_h = src_w; }
        else { dest_h = src_h; }

        if (dest_x >= dest_w || dest_y >= dest_h) return;

        int src_x, src_y;
        if (mode == 90) {      // Clockwise 90
            src_x = dest_y;
            src_y = src_w - 1 - dest_x;
        } else if (mode == 180) { // Clockwise 180
            src_x = src_w - 1 - dest_x;
            src_y = src_h - 1 - dest_y;
        } else {              // Clockwise 270
            src_x = src_h - 1 - dest_y;
            src_y = dest_x;
        }

        int dest_idx = (dest_y * dest_w + dest_x) * channels;
        int src_idx = (src_y * src_w + src_x) * channels;
        for (int c = 0; c < channels; c++) {
            dest[dest_idx + c] = src[src_idx + c];
        }
    }
    """)
    img = decode_image_base64(img_data)
    if not isinstance(img, np.ndarray): return img_data
    img = img.astype(np.uint8)
    src_h, src_w, channels = img.shape
    
    if degree_int == 90 or degree_int == 270:
        dest_h, dest_w = src_w, src_h
    else: # 180
        dest_h, dest_w = src_h, src_w

    src_gpu = cuda.mem_alloc(img.nbytes)
    dest_gpu = cuda.mem_alloc(dest_w * dest_h * channels)
    cuda.memcpy_htod(src_gpu, img)

    rotate_kernel = mod_rotate.get_function("lossless_rotate_kernel")
    block_dim = (16, 16, 1)
    grid_dim = ((dest_w + block_dim[0] - 1) // block_dim[0], (dest_h + block_dim[1] - 1) // block_dim[1])

    start_event = cuda.Event()
    end_event = cuda.Event()
    
    start_event.record()
    rotate_kernel(
        dest_gpu, src_gpu,
        np.int32(src_w), np.int32(src_h), np.int32(channels), np.int32(degree_int),
        block=block_dim, grid=grid_dim
    )
    end_event.record()
    end_event.synchronize()
    elapsed_ms = start_event.time_till(end_event)
    print(f"[PyCUDA] Lossless Transformation ({degree_int}) - Time: {elapsed_ms:.3f} ms")

    result_img = np.empty((dest_h, dest_w, channels), dtype=np.uint8)
    cuda.memcpy_dtoh(result_img, dest_gpu)
    
    src_gpu.free()
    dest_gpu.free()
    
    return nparray_to_pngbase64str(result_img)

# --- CUDA Kernel for Image Flip ---

def flip_pycuda(img_data: str, flip_type: str) -> str:
    """Flips an image using a custom PyCUDA kernel."""
    mod_flip = SourceModule("""
    __global__ void flip_kernel(unsigned char *dest, const unsigned char *src, int width, int height, int channels, int flip_x, int flip_y)
    {
            int dest_x = blockIdx.x * blockDim.x + threadIdx.x;
            int dest_y = blockIdx.y * blockDim.y + threadIdx.y;

            if (dest_x >= width || dest_y >= height) return;

            int src_x = flip_y ? (width - 1 - dest_x) : dest_x;
            int src_y = flip_x ? (height - 1 - dest_y) : dest_y;

            int dest_idx = (dest_y * width + dest_x) * channels;
            int src_idx = (src_y * width + src_x) * channels;

            for (int c = 0; c < channels; c++) dest[dest_idx + c] = src[src_idx + c];
        }
        """)
    img = decode_image_base64(img_data)
    if not isinstance(img, np.ndarray): return img_data
    img = img.astype(np.uint8)
    height, width, channels = img.shape

    flip_x = 'X' in flip_type
    flip_y = 'Y' in flip_type
    if not flip_x and not flip_y: return nparray_to_pngbase64str(img)

    img_gpu = cuda.mem_alloc(img.nbytes)
    dest_gpu = cuda.mem_alloc(img.nbytes)
    cuda.memcpy_htod(img_gpu, img)

    flip_kernel = mod_flip.get_function("flip_kernel")
    block_dim = (16, 16, 1)
    grid_dim = ((width + block_dim[0] - 1) // block_dim[0], (height + block_dim[1] - 1) // block_dim[1])

    start_time = time.perf_counter()
    flip_kernel(
        dest_gpu, img_gpu,
        np.int32(width), np.int32(height), np.int32(channels),
        np.int32(flip_x), np.int32(flip_y),
        block=block_dim, grid=grid_dim
    )
    cuda.Context.synchronize()
    end_time = time.perf_counter()
    print(f"[PyCUDA] flip({flip_type}) - Time: {(end_time - start_time) * 1000:.3f} ms")

    result_img = np.empty_like(img)
    cuda.memcpy_dtoh(result_img, dest_gpu)
    img_gpu.free()
    dest_gpu.free()
    return nparray_to_pngbase64str(result_img)

# --- CUDA Kernel for Gaussian Blur ---

def create_opencv_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    """
    Creates a 2D Gaussian kernel using OpenCV's standard method.
    This ensures the kernel is identical to the one used by cv2.GaussianBlur.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    kernel_1d = cv2.getGaussianKernel(kernel_size, sigma, ktype=cv2.CV_32F)
    kernel_2d = np.outer(kernel_1d, kernel_1d.transpose())
    return kernel_2d.astype(np.float32)

def apply_gaussian_blur_pycuda(img_data: str, kernel_size: int, sigma: float) -> str:
    """Applies Gaussian Blur using a custom PyCUDA kernel, corrected for consistency."""
    mod_gaussian_blur_color = SourceModule("""
    #include <math.h>

    __device__ int border_reflect_101(int x, int max_val) {
            if (x < 0) {
                return -x - 1;
            }
            if (x >= max_val) {
                return 2 * max_val - x - 2;
            }
            return x;
        }

        __global__ void gaussian_blur_kernel(unsigned char *dest, const unsigned char *src, const float* kernel, int width, int height, int channels, int kernel_size)
        {
            int g_x = blockIdx.x * blockDim.x + threadIdx.x;
            int g_y = blockIdx.y * blockDim.y + threadIdx.y;
            int kernel_radius = kernel_size / 2;

            if (g_x >= width || g_y >= height) return;

            float sum_b = 0.0f, sum_g = 0.0f, sum_r = 0.0f;

            for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
                for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
                    int sample_x = border_reflect_101(g_x + kx, width);
                    int sample_y = border_reflect_101(g_y + ky, height);
                    
                    float kernel_val = kernel[(ky + kernel_radius) * kernel_size + (kx + kernel_radius)];
                    int src_idx = (sample_y * width + sample_x) * channels;
                    
                    sum_b += (float)src[src_idx] * kernel_val;
                    sum_g += (float)src[src_idx + 1] * kernel_val;
                    sum_r += (float)src[src_idx + 2] * kernel_val;
                }
            }
            
            int dest_idx = (g_y * width + g_x) * channels;
            dest[dest_idx]     = (unsigned char)roundf(sum_b);
            dest[dest_idx + 1] = (unsigned char)roundf(sum_g);
            dest[dest_idx + 2] = (unsigned char)roundf(sum_r);
            
            if (channels > 3) {
                int center_idx = (g_y * width + g_x) * channels;
                dest[dest_idx + 3] = src[center_idx + 3];
            }
        }
        """)
    img = decode_image_base64(img_data)
    if not isinstance(img, np.ndarray): return img_data
    img = img.astype(np.uint8)
    height, width, channels = img.shape

    gaussian_kernel_cpu = create_opencv_gaussian_kernel(kernel_size, sigma)

    src_gpu = cuda.mem_alloc(img.nbytes)
    dest_gpu = cuda.mem_alloc(img.nbytes)
    kernel_gpu = cuda.mem_alloc(gaussian_kernel_cpu.nbytes)
    cuda.memcpy_htod(src_gpu, img)
    cuda.memcpy_htod(kernel_gpu, gaussian_kernel_cpu)

    gaussian_blur_kernel = mod_gaussian_blur_color.get_function("gaussian_blur_kernel")
    block_dim = (16, 16, 1)
    grid_dim = ((width + block_dim[0] - 1) // block_dim[0], (height + block_dim[1] - 1) // block_dim[1])

    start_event = cuda.Event()
    end_event = cuda.Event()

    start_event.record()
    gaussian_blur_kernel(
        dest_gpu, src_gpu, kernel_gpu,
        np.int32(width), np.int32(height), np.int32(channels), np.int32(kernel_size),
        block=block_dim, grid=grid_dim
    )
    end_event.record()
    end_event.synchronize()
    elapsed_ms = start_event.time_till(end_event)
    print(f"[PyCUDA] apply_gaussian_blur - Time: {elapsed_ms:.3f} ms")

    result_img = np.empty_like(img)
    cuda.memcpy_dtoh(result_img, dest_gpu)
    src_gpu.free()
    dest_gpu.free()
    kernel_gpu.free()
    return nparray_to_pngbase64str(result_img)

# --- CUDA Kernel for Mean Blur ---

def apply_mean_blur_pycuda(img_data: str, kernel_size: int) -> str:
    """
    Applies Mean Blur using a custom PyCUDA kernel.
    FINAL CORRECTED VERSION with rounding to match OpenCV's behavior.
    """
    mod_mean_blur = SourceModule("""
    #include <math.h>

    __global__ void mean_blur_kernel(unsigned char *dest, const unsigned char *src, int width, int height, int channels, int kernel_size)
    {
        int g_x = blockIdx.x * blockDim.x + threadIdx.x;
        int g_y = blockIdx.y * blockDim.y + threadIdx.y;
        int kernel_radius = kernel_size / 2;

        if (g_x >= width || g_y >= height) return;

        float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
        
        for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
            for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
                int sample_x = g_x + kx;
                int sample_y = g_y + ky;

                if (sample_x >= 0 && sample_x < width && sample_y >= 0 && sample_y < height) {
                    int idx = (sample_y * width + sample_x) * channels;
                    sum_b += src[idx];
                    sum_g += src[idx + 1];
                    sum_r += src[idx + 2];
                }
            }
        }

        int dest_idx = (g_y * width + g_x) * channels;
        
        float normalizer = 1.0f / (float)(kernel_size * kernel_size);

        dest[dest_idx]     = (unsigned char)roundf(sum_b * normalizer);
        dest[dest_idx + 1] = (unsigned char)roundf(sum_g * normalizer);
        dest[dest_idx + 2] = (unsigned char)roundf(sum_r * normalizer);

        if (channels == 4) {
            dest[dest_idx + 3] = src[dest_idx + 3];
        }
    }
    """)
    img = decode_image_base64(img_data)
    if not isinstance(img, np.ndarray): return img_data
    img = img.astype(np.uint8)
    height, width, channels = img.shape

    img_gpu = cuda.mem_alloc(img.nbytes)
    dest_gpu = cuda.mem_alloc(img.nbytes)
    cuda.memcpy_htod(img_gpu, img)

    mean_blur_kernel = mod_mean_blur.get_function("mean_blur_kernel")
    block_dim = (16, 16, 1)
    grid_dim = ((width + block_dim[0] - 1) // block_dim[0], (height + block_dim[1] - 1) // block_dim[1])

    start_event = cuda.Event()
    end_event = cuda.Event()

    start_event.record()
    mean_blur_kernel(
        dest_gpu, img_gpu,
        np.int32(width), np.int32(height), np.int32(channels), np.int32(kernel_size),
        block=block_dim, grid=grid_dim
    )
    end_event.record()
    end_event.synchronize()
    
    elapsed_ms = start_event.time_till(end_event)
    print(f"[PyCUDA] apply_mean_blur_pycuda - Time: {elapsed_ms:.3f} ms")

    result_img = np.empty_like(img)
    cuda.memcpy_dtoh(result_img, dest_gpu)
    
    img_gpu.free()
    dest_gpu.free()
    
    return nparray_to_pngbase64str(result_img)

# --- CUDA Kernel for Bilateral Filter ---

def apply_bilateral_filter_pycuda(img_data: str, diameter: int, sigmaColor: float, sigmaSpace: float) -> str:
    """
    Applies Bilateral Filter using a custom PyCUDA kernel.
    FINAL PERFECTED VERSION with circular neighborhood and rounding.
    """
    mod_bilateral_filter = SourceModule("""
    #include <math.h>

    __device__ int clamp(int x, int min_val, int max_val) {
            return min(max(x, min_val), max_val);
        }

        __device__ float gaussian(float x_sq, float sigma) {
            return expf(-x_sq / (2.0f * sigma * sigma));
        }

        __global__ void bilateral_filter_kernel(unsigned char *dest, const unsigned char *src, int width, int height, int channels, int diameter, float sigmaColor, float sigmaSpace)
        {
            int g_x = blockIdx.x * blockDim.x + threadIdx.x;
            int g_y = blockIdx.y * blockDim.y + threadIdx.y;
            int radius = diameter / 2;

            if (g_x >= width || g_y >= height) return;

            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
            float total_weight = 0.0f;

            int center_idx = (g_y * width + g_x) * channels;
            float center_b = src[center_idx];
            float center_g = src[center_idx + 1];
            float center_r = src[center_idx + 2];

            for (int ky = -radius; ky <= radius; ky++) {
                for (int kx = -radius; kx <= radius; kx++) {
                    if (kx * kx + ky * ky <= radius * radius) {
                        int sample_x = clamp(g_x + kx, 0, width - 1);
                        int sample_y = clamp(g_y + ky, 0, height - 1);

                        float space_dist_sq = (float)(kx * kx + ky * ky);
                        float space_weight = gaussian(space_dist_sq, sigmaSpace);

                        int sample_idx = (sample_y * width + sample_x) * channels;
                        float sample_b = src[sample_idx];
                        float sample_g = src[sample_idx + 1];
                        float sample_r = src[sample_idx + 2];

                        float color_dist_sq = (sample_b - center_b) * (sample_b - center_b) +
                                              (sample_g - center_g) * (sample_g - center_g) +
                                              (sample_r - center_r) * (sample_r - center_r);
                        float color_weight = gaussian(color_dist_sq, sigmaColor);

                        float weight = space_weight * color_weight;
                        
                        sum_b += sample_b * weight;
                        sum_g += sample_g * weight;
                        sum_r += sample_r * weight;
                        total_weight += weight;
                    }
                }
            }
            
            int dest_idx = (g_y * width + g_x) * channels;
            dest[dest_idx]     = (unsigned char)roundf(sum_b / total_weight);
            dest[dest_idx + 1] = (unsigned char)roundf(sum_g / total_weight);
            dest[dest_idx + 2] = (unsigned char)roundf(sum_r / total_weight);
            if (channels > 3) dest[dest_idx + 3] = src[center_idx + 3];
        }
        """)
    img = decode_image_base64(img_data)
    if not isinstance(img, np.ndarray): return img_data
    img = img.astype(np.uint8)
    height, width, channels = img.shape

    img_gpu = cuda.mem_alloc(img.nbytes)
    dest_gpu = cuda.mem_alloc(img.nbytes)
    cuda.memcpy_htod(img_gpu, img)

    bilateral_filter_kernel = mod_bilateral_filter.get_function("bilateral_filter_kernel")
    block_dim = (8, 8, 1)
    grid_dim = ((width + block_dim[0] - 1) // block_dim[0], (height + block_dim[1] - 1) // block_dim[1])

    start_event = cuda.Event()
    end_event = cuda.Event()
    start_event.record()
    bilateral_filter_kernel(
        dest_gpu, img_gpu,
        np.int32(width), np.int32(height), np.int32(channels),
        np.int32(diameter), np.float32(sigmaColor), np.float32(sigmaSpace),
        block=block_dim, grid=grid_dim
    )
    end_event.record()
    end_event.synchronize()
    elapsed_ms = start_event.time_till(end_event)
    print(f"[PyCUDA] apply_bilateral_filter - Time: {elapsed_ms:.3f} ms")

    result_img = np.empty_like(img)
    cuda.memcpy_dtoh(result_img, dest_gpu)
    img_gpu.free()
    dest_gpu.free()
    return nparray_to_pngbase64str(result_img)

# --- CUDA Kernels for Canny Edge Detection ---

def apply_canny_pycuda(img_data: str, th1: float, th2: float) -> str:
    """Applies Canny Edge Detection using custom PyCUDA kernels."""
    mod_canny = SourceModule("""
    #include <math.h>

    __global__ void sobel_filter_kernel(unsigned char *magnitude, float *direction, const unsigned char *src, int width, int height)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width - 1 || y >= height - 1 || x == 0 || y == 0) return;

            float Gx = (float)src[(y-1)*width + (x+1)] + 2.0f * (float)src[y*width + (x+1)] + (float)src[(y+1)*width + (x+1)] -
                       ((float)src[(y-1)*width + (x-1)] + 2.0f * (float)src[y*width + (x-1)] + (float)src[(y+1)*width + (x-1)]);

            float Gy = (float)src[(y+1)*width + (x-1)] + 2.0f * (float)src[(y+1)*width + x] + (float)src[(y+1)*width + (x+1)] -
                       ((float)src[(y-1)*width + (x-1)] + 2.0f * (float)src[(y-1)*width + x] + (float)src[(y-1)*width + (x+1)]);

            int idx = y * width + x;
            magnitude[idx] = (unsigned char)min(255.0f, fabsf(Gx) + fabsf(Gy));
            direction[idx] = atan2f(Gy, Gx);
        }

        __global__ void non_max_suppression_kernel(unsigned char *dest, const unsigned char *magnitude, const float *direction, int width, int height)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x >= width - 1 || y >= height - 1 || x == 0 || y == 0) return;

            int idx = y * width + x;
            float angle = direction[idx] * 180.0f / M_PI;
            if (angle < 0) angle += 180;

            unsigned char mag = magnitude[idx];
            unsigned char q = 255, r = 255;

            if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180)) {
                q = magnitude[idx+1]; r = magnitude[idx-1];
            } else if (22.5 <= angle && angle < 67.5) {
                q = magnitude[idx - width + 1]; r = magnitude[idx + width - 1];
            } else if (67.5 <= angle && angle < 112.5) {
                q = magnitude[idx - width]; r = magnitude[idx + width];
            } else if (112.5 <= angle && angle < 157.5) {
                q = magnitude[idx - width - 1]; r = magnitude[idx + width + 1];
            }

            dest[idx] = (mag >= q && mag >= r) ? mag : 0;
        }

        __global__ void double_threshold_kernel(unsigned char *dest, const unsigned char *src, int size, float low_threshold, float high_threshold)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;

            unsigned char val = src[idx];
            if (val >= high_threshold) dest[idx] = 255;
            else if (val >= low_threshold) dest[idx] = 25;
            else dest[idx] = 0;
        }

        __global__ void hysteresis_propagate_kernel(unsigned char *dest, const unsigned char *src, int width, int height)
        {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            if (x >= width || y >= height) return;
            int idx = y * width + x;

            unsigned char current_pixel = src[idx];
            dest[idx] = current_pixel;

            if (current_pixel == 25) {
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        if (i == 0 && j == 0) continue;
                        int nx = x + j;
                        int ny = y + i;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            if (src[ny * width + nx] == 255) {
                                dest[idx] = 255;
                                return;
                            }
                        }
                    }
                }
            }
        }

        __global__ void hysteresis_finalize_kernel(unsigned char *image, int size)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= size) return;

            if (image[idx] == 25) {
                image[idx] = 0;
            }
        }
        """)
    img = decode_image_base64(img_data)
    if not isinstance(img, np.ndarray): return img_data
    height, width, _ = img.shape

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 1.4)

    blurred_gpu = cuda.mem_alloc(blurred_img.nbytes)
    magnitude_gpu = cuda.mem_alloc(blurred_img.nbytes)
    direction_gpu = cuda.mem_alloc(blurred_img.astype(np.float32).nbytes)
    nms_gpu = cuda.mem_alloc(blurred_img.nbytes)
    threshold_gpu = cuda.mem_alloc(blurred_img.nbytes)
    cuda.memcpy_htod(blurred_gpu, blurred_img)

    sobel_kernel = mod_canny.get_function("sobel_filter_kernel")
    nms_kernel = mod_canny.get_function("non_max_suppression_kernel")
    threshold_kernel = mod_canny.get_function("double_threshold_kernel")
    hysteresis_propagate_kernel = mod_canny.get_function("hysteresis_propagate_kernel")
    hysteresis_finalize_kernel = mod_canny.get_function("hysteresis_finalize_kernel")

    block_dim_2d = (16, 16, 1)
    grid_dim_2d = ((width + block_dim_2d[0] - 1) // block_dim_2d[0], (height + block_dim_2d[1] - 1) // block_dim_2d[1])
    block_dim_1d = 256
    grid_dim_1d = (width * height + block_dim_1d - 1) // block_dim_1d

    start_time = time.perf_counter()
    sobel_kernel(magnitude_gpu, direction_gpu, blurred_gpu, np.int32(width), np.int32(height), block=block_dim_2d, grid=grid_dim_2d)
    nms_kernel(nms_gpu, magnitude_gpu, direction_gpu, np.int32(width), np.int32(height), block=block_dim_2d, grid=grid_dim_2d)
    threshold_kernel(threshold_gpu, nms_gpu, np.int32(width * height), np.float32(th1), np.float32(th2), block=(block_dim_1d,1,1), grid=(grid_dim_1d,1))
    
    buffer1_gpu = threshold_gpu
    buffer2_gpu = nms_gpu

    for _ in range(10):
        hysteresis_propagate_kernel(buffer2_gpu, buffer1_gpu, np.int32(width), np.int32(height), block=block_dim_2d, grid=grid_dim_2d)
        buffer1_gpu, buffer2_gpu = buffer2_gpu, buffer1_gpu
    
    final_propagated_gpu = buffer1_gpu

    hysteresis_finalize_kernel(final_propagated_gpu, np.int32(width * height), block=(block_dim_1d,1,1), grid=(grid_dim_1d,1))

    cuda.Context.synchronize()
    end_time = time.perf_counter()
    print(f"[PyCUDA] apply_canny - Time: {(end_time - start_time) * 1000:.3f} ms")

    result_img_gray = np.empty_like(gray_img)
    cuda.memcpy_dtoh(result_img_gray, final_propagated_gpu)

    blurred_gpu.free()
    magnitude_gpu.free()
    direction_gpu.free()
    nms_gpu.free()
    threshold_gpu.free()

    result_img_bgr = cv2.cvtColor(result_img_gray, cv2.COLOR_GRAY2BGR)
    return nparray_to_pngbase64str(result_img_bgr)

# --- CUDA Kernel for Warp Affine ---

def apply_warp_affine_pycuda(img_data: str, matrix_str: str) -> str:
    """Applies an affine transformation using a custom PyCUDA kernel with Bilinear Interpolation."""
    mod_warp_affine = SourceModule("""
    #include <math.h>

    __device__ unsigned char get_pixel_value(const unsigned char* src, int x, int y, int width, int height, int channel_offset, int channels) {
            if (x < 0 || x >= width || y < 0 || y >= height) {
                return 0;
            }
            return src[(y * width + x) * channels + channel_offset];
        }

        __global__ void warp_affine_bilinear_kernel(unsigned char* dest, const unsigned char* src, int width, int height, int channels, const float* M_inv)
        {
            int dest_x = blockIdx.x * blockDim.x + threadIdx.x;
            int dest_y = blockIdx.y * blockDim.y + threadIdx.y;

            if (dest_x >= width || dest_y >= height) return;

            float src_x_f = M_inv[0] * dest_x + M_inv[1] * dest_y + M_inv[2];
            float src_y_f = M_inv[3] * dest_x + M_inv[4] * dest_y + M_inv[5];

            int x1 = floorf(src_x_f);
            int y1 = floorf(src_y_f);
            int x2 = x1 + 1;
            int y2 = y1 + 1;

            float fx = src_x_f - x1;
            float fy = src_y_f - y1;

            int dest_idx = (dest_y * width + dest_x) * channels;

            for (int c = 0; c < channels; c++) {
                unsigned char p11 = get_pixel_value(src, x1, y1, width, height, c, channels);
                unsigned char p21 = get_pixel_value(src, x2, y1, width, height, c, channels);
                unsigned char p12 = get_pixel_value(src, x1, y2, width, height, c, channels);
                unsigned char p22 = get_pixel_value(src, x2, y2, width, height, c, channels);

                float top_interp = p11 * (1.0f - fx) + p21 * fx;
                float bottom_interp = p12 * (1.0f - fx) + p22 * fx;
                float final_value = top_interp * (1.0f - fy) + bottom_interp * fy;

                dest[dest_idx + c] = (unsigned char)roundf(final_value);
            }
        }
        """)
    img = decode_image_base64(img_data)
    if not isinstance(img, np.ndarray): return img_data
    img = img.astype(np.uint8)
    height, width, channels = img.shape

    vals = list(map(float, matrix_str.strip('[]').split(',')))
    M = np.array(vals, dtype=np.float32).reshape(2, 3)
    M_3x3 = np.vstack([M, [0, 0, 1]])
    try:
        M_inv_3x3 = np.linalg.inv(M_3x3)
        M_inv = M_inv_3x3[:2, :].flatten().astype(np.float32)
    except np.linalg.LinAlgError:
        print("Warning: Affine matrix is singular. Returning original image.")
        return nparray_to_pngbase64str(img)

    img_gpu = cuda.mem_alloc(img.nbytes)
    dest_gpu = cuda.mem_alloc(img.nbytes)
    matrix_gpu = cuda.mem_alloc(M_inv.nbytes)
    cuda.memcpy_htod(img_gpu, img)
    cuda.memcpy_htod(matrix_gpu, M_inv)

    warp_affine_kernel = mod_warp_affine.get_function("warp_affine_bilinear_kernel")
    block_dim = (16, 16, 1)
    grid_dim = ((width + block_dim[0] - 1) // block_dim[0], (height + block_dim[1] - 1) // block_dim[1])

    start_event = cuda.Event()
    end_event = cuda.Event()
    
    start_event.record()
    warp_affine_kernel(
        dest_gpu, img_gpu,
        np.int32(width), np.int32(height), np.int32(channels),
        matrix_gpu,
        block=block_dim, grid=grid_dim
    )
    end_event.record()
    end_event.synchronize()
    elapsed_ms = start_event.time_till(end_event)
    print(f"[PyCUDA] apply_warp_affine - Time: {elapsed_ms:.3f} ms")

    result_img = np.empty_like(img)
    cuda.memcpy_dtoh(result_img, dest_gpu)
    img_gpu.free()
    dest_gpu.free()
    matrix_gpu.free()
    return nparray_to_pngbase64str(result_img)

# --- Main Task Function ---

def Do_Task_CUDA(ImageRegion, MaskRegion, request):
    print('****************preprocessing (PyCUDA)**********************')
    preprocess_task = (request.form.getlist('preprocess_task')[0]).strip('[]').replace('"', '').split(',')
    
    with CudaContextManager():
        for task in preprocess_task:
            if not task: continue
            print(f"task is :{task}")
            if task == 'GammaCorrection':
                lut_value = float(request.form.getlist('lut_value')[0])
                ImageRegion = apply_lut_pycuda(ImageRegion, lut_value)
            elif task == 'TransformationDegrees':
                degree = request.form.getlist('degree')[0]
                ImageRegion = transformation_degrees_pycuda(ImageRegion, degree)
                if MaskRegion: MaskRegion = transformation_degrees_pycuda(MaskRegion, degree)
            elif task == 'TransformationFlip':
                flip_type = request.form.getlist('flip_type')[0]
                ImageRegion = flip_pycuda(ImageRegion, flip_type)
                if MaskRegion: MaskRegion = flip_pycuda(MaskRegion, flip_type)
            elif task == 'Guassian':
                gaussian_blur_size = int(request.form.getlist('gaussian_blur_size')[0])
                gaussian_blur_sigma = float(request.form.getlist('gaussian_blur_sigma')[0])
                ImageRegion = apply_gaussian_blur_pycuda(ImageRegion, gaussian_blur_size, gaussian_blur_sigma)
            elif task == 'Blur':
                ksize = int(request.form.getlist('blur_size')[0])
                ImageRegion = apply_mean_blur_pycuda(ImageRegion, ksize)
            elif task == 'BilateralFilter':
                diameter = int(request.form.getlist('diameter')[0])
                sigmaColor = float(request.form.getlist('sigmaColor')[0])
                sigmaSpace = float(request.form.getlist('sigmaSpace')[0])
                ImageRegion = apply_bilateral_filter_pycuda(ImageRegion, diameter, sigmaColor, sigmaSpace)
            elif task == 'Canny':
                th1 = float(request.form.getlist('threshold1')[0])
                th2 = float(request.form.getlist('threshold2')[0])
                ImageRegion = apply_canny_pycuda(ImageRegion, th1, th2)
            elif task == 'WarpAffine':
                matrix_str = request.form.getlist('affine_matrix')[0]
                ImageRegion = apply_warp_affine_pycuda(ImageRegion, matrix_str)
            else:
                print(f"Task '{task}' is not a recognized CUDA function.")

    return ([ImageRegion], [MaskRegion])