import base64
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import time

def is_cuda_available():
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except:
        return False

CUDA_AVAILABLE = is_cuda_available() & True
print("CUDA available:", CUDA_AVAILABLE)

def decode_image_base64(img_data: str):
    byte_arr = base64.b64decode(img_data)
    img_np = np.frombuffer(byte_arr, np.uint8)
    return cv2.imdecode(img_np, cv2.IMREAD_COLOR)

def nparray_to_pngbase64str(img: np.ndarray):
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def Do_Task(ImageRegion, MaskRegion, request):
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
        # --- GPU 路徑 (使用 transpose 和 flip 實現無損旋轉) ---
        gpu_bgr = cv2.cuda_GpuMat()
        gpu_bgr.upload(img)
        
        # *** 關鍵修正：先轉換成 BGRA 格式以滿足 transpose 函數的要求 ***
        gpu_rgba = cv2.cuda.cvtColor(gpu_bgr, cv2.COLOR_BGR2BGRA)
        
        start = cv2.cuda.Event()
        end = cv2.cuda.Event()
        start.record()

        if degree_int == 90:
            # 對 4 通道的 BGRA 影像進行操作
            gpu_transposed = cv2.cuda.transpose(gpu_rgba)
            gpu_rotated_rgba = cv2.cuda.flip(gpu_transposed, 1)
        elif degree_int == 180:
            # flip 函數通常支援更多格式，但為保持程式碼一致性，我們也對 BGRA 操作
            gpu_rotated_rgba = cv2.cuda.flip(gpu_rgba, -1)
        else: # 270
            # 對 4 通道的 BGRA 影像進行操作
            gpu_transposed = cv2.cuda.transpose(gpu_rgba)
            gpu_rotated_rgba = cv2.cuda.flip(gpu_transposed, 0)
            
        # *** 關鍵修正：將結果轉換回 BGR 格式 ***
        gpu_rotated_bgr = cv2.cuda.cvtColor(gpu_rotated_rgba, cv2.COLOR_BGRA2BGR)
            
        end.record()
        end.waitForCompletion()
        elapsed = cv2.cuda.Event.elapsedTime(start, end)
        print(f"[GPU] Lossless Rotation ({degree}) - Time: {elapsed:.3f} ms")
        
        result_mat = gpu_rotated_bgr.download()

    else:
        # --- CPU 路徑 (使用 cv2.rotate 實現無損旋轉) ---
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
        # 1. 上传到 GPU 并扩为 4 通道
        gpu_bgr  = cv2.cuda_GpuMat(); gpu_bgr.upload(img)
        gpu_rgba = cv2.cuda.cvtColor(gpu_bgr, cv2.COLOR_BGR2BGRA)

        # 2. 根据 flip_type 构造仿射矩阵
        if flip_type == 'X':       # 水平翻转（沿 X 轴）
            M = np.array([[1,  0, 0],
                          [0, -1, h-1]], dtype=np.float32)
        elif flip_type == 'Y':     # 垂直翻转（沿 Y 轴）
            M = np.array([[-1, 0, w-1],
                          [ 0, 1, 0]], dtype=np.float32)
        elif flip_type == 'XY':    # 同时翻转
            M = np.array([[-1, 0, w-1],
                          [0, -1, h-1]], dtype=np.float32)
        else:
            # 不翻转
            return nparray_to_pngbase64str(img)

        # 3. 调用 CUDA warpAffine
        start = cv2.cuda.Event(); end = cv2.cuda.Event()
        start.record()
        gpu_flipped_rgba = cv2.cuda.warpAffine(
            gpu_rgba, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )  # :contentReference[oaicite:0]{index=0}
        end.record(); end.waitForCompletion()

        # （可选）打印耗时
        elapsed = cv2.cuda.Event.elapsedTime(start, end)
        print(f"[GPU] flip({flip_type}) - Time: {elapsed:.3f} ms")

        # 4. 转回 BGR 并下载
        gpu_flipped_bgr = cv2.cuda.cvtColor(gpu_flipped_rgba, cv2.COLOR_BGRA2BGR)
        result = gpu_flipped_bgr.download()
    else:
        t0 = time.perf_counter()

        # CPU 回退
        if flip_type == 'X':
            result = cv2.flip(img, 0)
        elif flip_type == 'Y':
            result = cv2.flip(img, 1)
        elif flip_type == 'XY':
            result = cv2.flip(img, -1)
        else:
            result = img

        # CPU 計時結束
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        print(f"[CPU] flip({flip_type}) - Time: {elapsed_ms:.3f} ms")

    return nparray_to_pngbase64str(result)

def apply_lut(img_data: str, lut_value: float) -> str:
    img = decode_image_base64(img_data)
    lut = np.array([np.clip(pow(i / 255.0, lut_value) * 255.0, 0, 255)
                    for i in range(256)], dtype=np.uint8).reshape((256, 1))
    if CUDA_AVAILABLE:
        # 準備輸入
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)

        # 建立事件
        start = cv2.cuda.Event()
        end   = cv2.cuda.Event()

        # 記錄開始
        start.record()

        # 執行 CUDA

        # 在 host 端準備 LUT（dtype 必須是 uint8、長度 256）
        lut_array = np.asarray(lut, dtype=np.uint8)

        # 建立 CUDA LookUpTable filter
        lut_filter = cv2.cuda.createLookUpTable(lut_array)
        gpu_result = lut_filter.transform(gpu_img)

        # 記錄結束
        end.record()

        # 等待這個結束事件完成
        end.waitForCompletion()

        # 計算時間
        elapsed_time = cv2.cuda.Event.elapsedTime(start, end)
        print(f"[GPU] apply_lut - Time: {elapsed_time:.3f} ms")
        result = gpu_result.download()

    else:
        t0 = time.perf_counter()

        result = cv2.LUT(img, lut)
        
        # CPU 計時結束
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        print(f"[CPU] apply_lut - Time: {elapsed_ms:.3f} ms")

    return nparray_to_pngbase64str(result)

def apply_gaussian_blur(img_data: str, kernel_size: int, sigma: float) -> str:
    img = decode_image_base64(img_data)
    if CUDA_AVAILABLE:
        # 準備輸入
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)

        # 建立事件
        start = cv2.cuda.Event()
        end   = cv2.cuda.Event()

        # 記錄開始
        start.record()

        # 執行 CUDA Gaussian filter
        gaussian_filter = cv2.cuda.createGaussianFilter(
            cv2.CV_8UC3, cv2.CV_8UC3, (kernel_size, kernel_size), sigma)
        gpu_result = gaussian_filter.apply(gpu_img)

        # 記錄結束
        end.record()

        # 等待這個結束事件完成
        end.waitForCompletion()

        # 計算時間
        elapsed_time = cv2.cuda.Event.elapsedTime(start, end)
        print(f"[GPU] apply_gaussian_blur - Time: {elapsed_time:.3f} ms")
        result = gpu_result.download()
    else:
        t0 = time.perf_counter()
        result = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        
        # CPU 計時結束
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        print(f"[CPU] apply_gaussian_blur - Time: {elapsed_ms:.3f} ms")

    return nparray_to_pngbase64str(result)

def apply_mean_blur(img_data: str, kernel_size: int) -> str:
    img = decode_image_base64(img_data)
    if CUDA_AVAILABLE:
        # 1. 準備輸入
        gpu_img_bgr = cv2.cuda_GpuMat()
        gpu_img_bgr.upload(img)
        
        # 2. BGR -> BGRA (3-channel to 4-channel) to meet the filter's requirement
        gpu_img_bgra = cv2.cuda.cvtColor(gpu_img_bgr, cv2.COLOR_BGR2BGRA)
        
        # 建立與 CPU 相同的核心
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

        # 建立事件以進行計時
        start = cv2.cuda.Event()
        end   = cv2.cuda.Event()

        # 記錄開始
        start.record()

        # 3. 建立 LinearFilter，指定處理 4 通道影像 (CV_8UC4) 
        #    並使用正確的 'borderMode' 參數
        filter = cv2.cuda.createLinearFilter(
            srcType=cv2.CV_8UC4, 
            dstType=cv2.CV_8UC4, 
            kernel=kernel, 
            borderMode=cv2.BORDER_CONSTANT
        )
        # 4. 將濾鏡應用在 4 通道的影像上
        gpu_result_bgra = filter.apply(gpu_img_bgra)
        
        # 5. BGRA -> BGR (4-channel back to 3-channel) for consistent output
        gpu_result_bgr = cv2.cuda.cvtColor(gpu_result_bgra, cv2.COLOR_BGRA2BGR)

        # 記錄結束並等待
        end.record()
        end.waitForCompletion()
        
        # 計算時間
        elapsed_time = cv2.cuda.Event.elapsedTime(start, end)
        print(f"[OpenCV CUDA] apply_mean_blur_opencv_cuda - Time: {elapsed_time:.3f} ms")
        
        # 6. 下載最終的 3 通道影像
        result = gpu_result_bgr.download()
        
    else:
        t0 = time.perf_counter()
        # To achieve perfect consistency with the custom CUDA kernel,
        # use filter2D which gives more control over border handling.
        # The custom CUDA kernels effectively use BORDER_CONSTANT with a value of 0.
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        result = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)

        # CPU 計時結束
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        print(f"[CPU] apply_mean_blur - Time: {elapsed_ms:.3f} ms")

    return nparray_to_pngbase64str(result)

def apply_canny(img_data: str, th1: float, th2: float) -> str:
    img = decode_image_base64(img_data)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if CUDA_AVAILABLE:
        # 1. 上傳影像到 GPU
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)

        # 2. 轉成灰階（GPU 端）
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)

        # 3. 建立事件
        start = cv2.cuda.Event()
        end   = cv2.cuda.Event()

        # 4. 記錄開始
        start.record()

        # 5. 執行 CUDA Canny（輸入必須是 CV_8UC1）
        canny_detector = cv2.cuda.createCannyEdgeDetector(th1, th2)
        gpu_edges = canny_detector.detect(gpu_gray)

        # 6. 記錄結束並等待完成
        end.record()
        end.waitForCompletion()

        # 7. 計算耗時
        elapsed_time = cv2.cuda.Event.elapsedTime(start, end)
        print(f"[GPU] apply_canny - Time: {elapsed_time:.3f} ms")

        # 8. 下載結果並命名為 edges
        edges = gpu_edges.download()

        # 9. 轉成 BGR（如果後續需要三通道）
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    else:
        t0 = time.perf_counter()

        edges = cv2.Canny(gray, th1, th2)
        
        # CPU 計時結束
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000
        print(f"[CPU] apply_canny - Time: {elapsed_ms:.3f} ms")

    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return nparray_to_pngbase64str(edges_bgr)

def apply_bilateral_filter(img_data: str, diameter: int, sigmaColor: float, sigmaSpace: float) -> str:
    img = decode_image_base64(img_data)

    # 為了與 PyCUDA 版本統一，明確指定邊界處理模式
    border_mode = cv2.BORDER_REPLICATE

    if CUDA_AVAILABLE:
        gpu_img = cv2.cuda_GpuMat()
        gpu_img.upload(img)

        start = cv2.cuda.Event()
        end   = cv2.cuda.Event()
        start.record()

        # *** 關鍵修正：傳入 borderMode 參數 ***
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
        print(f"[GPU] apply_bilateral_filter - Time: {elapsed:.3f} ms")
        result_mat = gpu_result.download()

    else:
        t0 = time.perf_counter()
        # *** 關鍵修正：傳入 borderMode 參數 ***
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

    # 解析仿射矩陣
    vals = list(map(float, matrix_str.strip('[]').split(',')))
    M64 = np.array(vals, dtype=np.float32).reshape(2, 3)

    # 統一參數
    interpolation_flags = cv2.INTER_LINEAR
    border_mode = cv2.BORDER_CONSTANT
    border_value = (0, 0, 0, 0) # 預設邊界顏色為黑

    if CUDA_AVAILABLE:
        gpu_bgr = cv2.cuda_GpuMat()
        gpu_bgr.upload(img)
        # GPU 的 warpAffine 對 4 通道有更好的性能
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
        print(f"[GPU] apply_warp_affine - Time: {elapsed:.3f} ms")

        gpu_rot_bgr = cv2.cuda.cvtColor(gpu_rot_rgba, cv2.COLOR_BGRA2BGR)
        result_mat = gpu_rot_bgr.download()
    else:
        t0 = time.perf_counter()
        # *** 關鍵修正：明確傳入所有參數以保證行為一致 ***
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
