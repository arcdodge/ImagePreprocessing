import base64
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import time

CUDA_AVAILABLE = False

def decode_image_base64(img_data: str):
    byte_arr = base64.b64decode(img_data)
    img_np = np.frombuffer(byte_arr, np.uint8)
    return cv2.imdecode(img_np, cv2.IMREAD_COLOR)

def nparray_to_pngbase64str(img: np.ndarray):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def Do_Task(ImageRegion, MaskRegion, request):
    global CUDA_AVAILABLE
    acceleration = request.form.get('acceleration', 'CPU')
    
    if acceleration == 'CUDA' and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        CUDA_AVAILABLE = True
    elif acceleration == 'CVCUDA':
        CUDA_AVAILABLE = True
    else:
        CUDA_AVAILABLE = False
        
    print(f"Acceleration mode: {acceleration}, CUDA_AVAILABLE: {CUDA_AVAILABLE}")

    print('****************preprocessing**********************')
    preprocess_task = (request.form.getlist('preprocess_task')[0]).strip('[]').replace('"', '').split(',')
    if len(preprocess_task) > 0:
        for task in preprocess_task:
            print(f"task is :{task}")
            if task == 'TransformationDegrees':
                degree = request.form.getlist('degree')[0]
                ImageRegion = transformation_degrees(ImageRegion, degree)
                if MaskRegion != '':
                    MaskRegion = transformation_degrees(MaskRegion, degree)
            elif task == 'TransformationFlip':
                flip_type = request.form.getlist('flip_type')[0]
                ImageRegion = flip(ImageRegion, flip_type)
                if MaskRegion != '':
                    MaskRegion = flip(MaskRegion, flip_type)
            elif task == 'GammaCorrection':
                lut_value = float(request.form.getlist('lut_value')[0])
                ImageRegion = apply_lut(ImageRegion, lut_value)
            elif task == 'Guassian':
                gaussian_blur_size = int(request.form.getlist('gaussian_blur_size')[0])
                gaussian_blur_sigma = float(request.form.getlist('gaussian_blur_sigma')[0])
                ImageRegion = apply_gaussian_blur(ImageRegion, gaussian_blur_size, gaussian_blur_sigma)
            elif task == 'Blur':
                ksize = int(request.form.getlist('blur_size')[0])
                ImageRegion = apply_mean_blur(ImageRegion, ksize)
            elif task == 'BilateralFilter':
                diameter = int(request.form.getlist('diameter')[0])
                sigmaColor = float(request.form.getlist('sigmaColor')[0])
                sigmaSpace = float(request.form.getlist('sigmaSpace')[0])
                ImageRegion = apply_bilateral_filter(ImageRegion, diameter, sigmaColor, sigmaSpace)
            elif task == 'Canny':
                th1 = float(request.form.getlist('threshold1')[0])
                th2 = float(request.form.getlist('threshold2')[0])
                ImageRegion = apply_canny(ImageRegion, th1, th2)
            elif task == 'WarpAffine':
                matrix_str = request.form.getlist('affine_matrix')[0]
                ImageRegion = apply_warp_affine(ImageRegion, matrix_str)
            else:
                ImageRegion = ImageRegion

        return ([ImageRegion], [MaskRegion])

def transformation_degrees(img_data: str, degree: float) -> str:
    """
    Performs a LOSSLESS 90/180/270 degree rotation, ensuring perfect consistency
    between the CPU (cv2.rotate) and GPU (cv2.cuda.transpose/flip) paths.
    """
    img = decode_image_base64(img_data)
    if not isinstance(img, np.ndarray):
        return img_data
        
    degree_int = int(float(degree))
    
    if degree_int not in (90, 180, 270):
        return nparray_to_pngbase64str(img)

    if CUDA_AVAILABLE:
        gpu_bgr = cv2.cuda_GpuMat()
        gpu_bgr.upload(img)
        
        gpu_rgba = cv2.cuda.cvtColor(gpu_bgr, cv2.COLOR_BGR2BGRA)
        
        start = cv2.cuda.Event()
        end = cv2.cuda.Event()
        start.record()

        if degree_int == 90:
            gpu_transposed = cv2.cuda.transpose(gpu_rgba)
            gpu_rotated_rgba = cv2.cuda.flip(gpu_transposed, 1)
        elif degree_int == 180:
            gpu_rotated_rgba = cv2.cuda.flip(gpu_rgba, -1)
        else: # 270
            gpu_transposed = cv2.cuda.transpose(gpu_rgba)
            gpu_rotated_rgba = cv2.cuda.flip(gpu_transposed, 0)
            
        gpu_rotated_bgr = cv2.cuda.cvtColor(gpu_rotated_rgba, cv2.COLOR_BGRA2BGR)
            
        end.record()
        end.waitForCompletion()
        elapsed = cv2.cuda.Event.elapsedTime(start, end)
        print(f"[OpenCV CUDA] Lossless Rotation ({degree}) - Time: {elapsed:.3f} ms")
        
        result_mat = gpu_rotated_bgr.download()

    else:
        t0 = time.perf_counter()
        
        if degree_int == 90:
            rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif degree_int == 180:
            rotated_img = cv2.rotate(img, cv2.ROTATE_180)
        else: # 270
            rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
        result_mat = rotated_img
        
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        print(f"[CPU] Lossless Rotation ({degree}) - Time: {elapsed_ms:.3f} ms")

    return nparray_to_pngbase64str(result_mat)

def flip(img_data: str, flip_type: str) -> str:
    img = decode_image_base64(img_data)
    h, w = img.shape[:2]

    if CUDA_AVAILABLE:
        gpu_bgr  = cv2.cuda_GpuMat(); gpu_bgr.upload(img)
        gpu_rgba = cv2.cuda.cvtColor(gpu_bgr, cv2.COLOR_BGR2BGRA)

        if flip_type == 'X':
            M = np.array([[1,  0, 0],
                          [0, -1, h-1]], dtype=np.float32)
        elif flip_type == 'Y':
            M = np.array([[-1, 0, w-1],
                          [ 0, 1, 0]], dtype=np.float32)
        elif flip_type == 'XY':
            M = np.array([[-1, 0, w-1],
                          [0, -1, h-1]], dtype=np.float32)
        else:
            return nparray_to_pngbase64str(img)

        start = cv2.cuda.Event(); end = cv2.cuda.Event()
        start.record()
        gpu_flipped_rgba = cv2.cuda.warpAffine(
            gpu_rgba, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        end.record(); end.waitForCompletion()

        elapsed = cv2.cuda.Event.elapsedTime(start, end)
        print(f"[OpenCV CUDA] flip({flip_type}) - Time: {elapsed:.3f} ms")

        gpu_flipped_bgr = cv2.cuda.cvtColor(gpu_flipped_rgba, cv2.COLOR_BGRA2BGR)
        result = gpu_flipped_bgr.download()
    else:
        t0 = time.perf_counter()

        if flip_type == 'X':
            result = cv2.flip(img, 0)
        elif flip_type == 'Y':
            result = cv2.flip(img, 1)
        elif flip_type == 'XY':
            result = cv2.flip(img, -1)
        else:
            result = img

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        print(f"[CPU] flip({flip_type}) - Time: {elapsed_ms:.3f} ms")

    return nparray_to_pngbase64str(result)

def apply_lut(img_data: str, lut_value: float) -> str:
    img = decode_image_base64(img_data)
    lut = np.array([np.clip(pow(i / 255.0, lut_value) * 255.0, 0, 255)
                    for i in range(256)], dtype=np.uint8).reshape((256, 1))
    if CUDA_AVAILABLE:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)

        start = cv2.cuda.Event()
        end   = cv2.cuda.Event()

        start.record()

        lut_array = np.asarray(lut, dtype=np.uint8)

        lut_filter = cv2.cuda.createLookUpTable(lut_array)
        gpu_result = lut_filter.transform(gpu_img)

        end.record()

        end.waitForCompletion()

        elapsed_time = cv2.cuda.Event.elapsedTime(start, end)
        print(f"[OpenCV CUDA] apply_lut - Time: {elapsed_time:.3f} ms")
        result = gpu_result.download()

    else:
        t0 = time.perf_counter()

        result = cv2.LUT(img, lut)
        
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        print(f"[CPU] apply_lut - Time: {elapsed_ms:.3f} ms")

    return nparray_to_pngbase64str(result)

def apply_gaussian_blur(img_data: str, kernel_size: int, sigma: float) -> str:
    img = decode_image_base64(img_data)
    if CUDA_AVAILABLE:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)

        start = cv2.cuda.Event()
        end   = cv2.cuda.Event()

        start.record()

        gaussian_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC3, cv2.CV_8UC3, (kernel_size, kernel_size), sigma)
        gpu_result = gaussian_filter.apply(gpu_img)

        end.record()
        end.waitForCompletion()

        elapsed_time = cv2.cuda.Event.elapsedTime(start, end)
        print(f"[OpenCV CUDA] apply_gaussian_blur - Time: {elapsed_time:.3f} ms")
        result = gpu_result.download()
    else:
        t0 = time.perf_counter()
        result = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        print(f"[CPU] apply_gaussian_blur - Time: {elapsed_ms:.3f} ms")

    return nparray_to_pngbase64str(result)

def apply_mean_blur(img_data: str, kernel_size: int) -> str:
    img = decode_image_base64(img_data)
    if CUDA_AVAILABLE:
        gpu_img_bgr = cv2.cuda_GpuMat()
        gpu_img_bgr.upload(img)
        gpu_img_bgra = cv2.cuda.cvtColor(gpu_img_bgr, cv2.COLOR_BGR2BGRA)
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

        start = cv2.cuda.Event()
        end   = cv2.cuda.Event()

        start.record()
        filter = cv2.cuda.createLinearFilter(
            srcType=cv2.CV_8UC4, 
            dstType=cv2.CV_8UC4, 
            kernel=kernel, 
            borderMode=cv2.BORDER_CONSTANT
        )
        gpu_result_bgra = filter.apply(gpu_img_bgra)
        
        gpu_result_bgr = cv2.cuda.cvtColor(gpu_result_bgra, cv2.COLOR_BGRA2BGR)

        end.record()
        end.waitForCompletion()
        
        elapsed_time = cv2.cuda.Event.elapsedTime(start, end)
        print(f"[OpenCV CUDA] apply_mean_blur_opencv_cuda - Time: {elapsed_time:.3f} ms")
        
        result = gpu_result_bgr.download()
        
    else:
        t0 = time.perf_counter()
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        result = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        print(f"[CPU] apply_mean_blur - Time: {elapsed_ms:.3f} ms")

    return nparray_to_pngbase64str(result)

def apply_canny(img_data: str, th1: float, th2: float) -> str:
    img = decode_image_base64(img_data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if CUDA_AVAILABLE:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)

        start = cv2.cuda.Event()
        end   = cv2.cuda.Event()
        start.record()

        canny_detector = cv2.cuda.createCannyEdgeDetector(th1, th2)
        gpu_edges = canny_detector.detect(gpu_gray)

        end.record()
        end.waitForCompletion()

        elapsed_time = cv2.cuda.Event.elapsedTime(start, end)
        print(f"[OpenCV CUDA] apply_canny - Time: {elapsed_time:.3f} ms")
        edges = gpu_edges.download()
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    else:
        t0 = time.perf_counter()
        edges = cv2.Canny(gray, th1, th2)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        print(f"[CPU] apply_canny - Time: {elapsed_ms:.3f} ms")

    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return nparray_to_pngbase64str(edges_bgr)

def apply_bilateral_filter(img_data: str, diameter: int, sigmaColor: float, sigmaSpace: float) -> str:
    img = decode_image_base64(img_data)

    border_mode = cv2.BORDER_REPLICATE

    if CUDA_AVAILABLE:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)

        start = cv2.cuda.Event()
        end   = cv2.cuda.Event()
        start.record()

        gpu_result = cv2.cuda.bilateralFilter(
            gpu_img,
            diameter,
            sigmaColor,
            sigmaSpace,
            borderMode=border_mode
        )

        end.record()
        end.waitForCompletion()
        elapsed = cv2.cuda.Event.elapsedTime(start, end)
        print(f"[OpenCV CUDA] apply_bilateral_filter - Time: {elapsed:.3f} ms")
        result_mat = gpu_result.download()

    else:
        t0 = time.perf_counter()
        result_mat = cv2.bilateralFilter(
            img,
            diameter,
            sigmaColor,
            sigmaSpace,
            borderType=border_mode
        )
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        print(f"[CPU] apply_bilateral_filter - Time: {elapsed_ms:.3f} ms")

    return nparray_to_pngbase64str(result_mat)

def apply_warp_affine(img_data: str, matrix_str: str) -> str:
    img = decode_image_base64(img_data)
    h, w = img.shape[:2]

    vals = list(map(float, matrix_str.strip('[]').split(',')))
    M64 = np.array(vals, dtype=np.float32).reshape(2, 3)

    interpolation_flags = cv2.INTER_LINEAR
    border_mode = cv2.BORDER_CONSTANT
    border_value = (0, 0, 0, 0) # 預設邊界顏色為黑

    if CUDA_AVAILABLE:
        gpu_bgr = cv2.cuda_GpuMat()
        gpu_bgr.upload(img)
        gpu_rgba = cv2.cuda.cvtColor(gpu_bgr, cv2.COLOR_BGR2BGRA)

        start = cv2.cuda.Event()
        end = cv2.cuda.Event()
        start.record()

        gpu_rot_rgba = cv2.cuda.warpAffine(
            gpu_rgba, M64, (w, h),
            flags=interpolation_flags,
            borderMode=border_mode,
            borderValue=border_value
        )

        end.record()
        end.waitForCompletion()
        elapsed = cv2.cuda.Event.elapsedTime(start, end)
        print(f"[OpenCV CUDA] apply_warp_affine - Time: {elapsed:.3f} ms")

        gpu_rot_bgr = cv2.cuda.cvtColor(gpu_rot_rgba, cv2.COLOR_BGRA2BGR)
        result_mat = gpu_rot_bgr.download()
    else:
        t0 = time.perf_counter()
        result_mat = cv2.warpAffine(
            img, M64, (w, h),
            flags=interpolation_flags,
            borderMode=border_mode,
            borderValue=border_value
        )
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        print(f"[CPU] apply_warp_affine - Time: {elapsed_ms:.3f} ms")

    return nparray_to_pngbase64str(result_mat)
