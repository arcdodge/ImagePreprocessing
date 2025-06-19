import os
import numpy as np
import cv2
import time
import base64
from unittest.mock import MagicMock

# --- 設定您的新版 CUDA Toolkit bin 目錄路徑 ---
# 請將此路徑替換為您系統上正確的路徑
# 例如: 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin' (Windows)
# 或 '/usr/local/cuda-12.2/bin' (Linux)
YOUR_CUDA_BIN_PATH = "/usr/local/cuda-12.9/bin"

# 將此路徑加到環境變數 PATH 的最前面
if os.path.exists(YOUR_CUDA_BIN_PATH):
    os.environ['PATH'] = YOUR_CUDA_BIN_PATH + os.pathsep + os.environ.get('PATH', '')
    print(f"訊息: 已將 {YOUR_CUDA_BIN_PATH} 加入到 PATH 環境變數。")
else:
    print(f"警告: 指定的 CUDA 路徑不存在: {YOUR_CUDA_BIN_PATH}")
    print("      PyCUDA 可能會使用系統預設的舊版 nvcc。")

# 匯入要測試的模組
import preprocessing
import preprocessing_cuda

def create_mock_request(task_name, params):
    """建立一個模擬的 Flask request 物件"""
    # getlist 應該為每個鍵返回一個值列表
    form_data = {'preprocess_task': [f'["{task_name}"]']}
    for key, value in params.items():
        form_data[key] = [str(value)]
    
    request = MagicMock()
    request.form.getlist.side_effect = lambda k: form_data.get(k, [])
    return request

def nparray_to_pngbase64str(img: np.ndarray):
    """將 numpy 陣列編碼為 PNG base64 字串"""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def decode_image_base64(img_data: str):
    """將 base64 字串解碼為 BGR numpy 陣列"""
    if isinstance(img_data, list) and len(img_data) > 0:
        img_data = img_data[0] # 從列表中取出 base64 字串
    byte_arr = base64.b64decode(img_data)
    img_np = np.frombuffer(byte_arr, np.uint8)
    return cv2.imdecode(img_np, cv2.IMREAD_COLOR)

def compare_images(img1, img2, label1="Image 1", label2="Image 2"):
    """比較兩個影像是否相同，並回報差異"""
    if img1.shape != img2.shape:
        print(f"  - ❌ 尺寸不匹配: {label1} shape {img1.shape} vs {label2} shape {img2.shape}")
        return False
    
    diff = cv2.absdiff(img1, img2)
    non_zero_count = np.count_nonzero(diff)
    
    if non_zero_count == 0:
        print(f"  - ✅ 圖像完全一致: {label1} vs {label2}")
        return True
    else:
        total_pixels = diff.shape[0] * diff.shape[1] * diff.shape[2]
        diff_percentage = (non_zero_count / total_pixels) * 100
        mean_diff = np.mean(diff)
        print(f"  - ⚠️ 圖像存在差異: {label1} vs {label2}")
        print(f"    - 差異像素數量: {non_zero_count} / {total_pixels} ({diff_percentage:.4f}%)")
        print(f"    - 平均差異值: {mean_diff:.4f}")
        return False

def run_benchmark(task_name, params):
    """執行指定任務的效能測試與比較"""
    print(f"\n{'='*20} 測試任務: {task_name} {'='*20}")
    print(f"參數: {params}")

    # 1. 準備輸入資料
    # 建立一張 1024x768 的漸層影像
    width, height = 500, 500
    img_cpu = np.zeros((height, width, 3), dtype=np.uint8)
    img_cpu[:, :, 0] = np.tile(np.linspace(0, 255, width), (height, 1)) # Blue channel gradient
    img_cpu[:, :, 1] = np.tile(np.linspace(0, 255, height).reshape(height, 1), (1, width)) # Green channel gradient
    img_cpu[:, :, 2] = 128 # Red channel constant
    
    img_base64 = nparray_to_pngbase64str(img_cpu)
    mock_request = create_mock_request(task_name, params)

    # --- 執行並計時 ---
    results = {}
    times = {}

    # 2. CPU 版本
    print("\n--- 執行 CPU 版本 ---")
    preprocessing.CUDA_AVAILABLE = False # 強制使用 CPU
    start_time = time.perf_counter()
    result_cpu_base64, _ = preprocessing.Do_Task(img_base64, '', mock_request)
    times['CPU'] = (time.perf_counter() - start_time) * 1000
    results['CPU'] = decode_image_base64(result_cpu_base64)
    print(f"耗時: {times['CPU']:.3f} ms")

    # 3. OpenCV CUDA 版本
    print("\n--- 執行 OpenCV CUDA 版本 ---")
    preprocessing.CUDA_AVAILABLE = True # 強制使用 OpenCV CUDA
    start_time = time.perf_counter()
    result_cv_cuda_base64, _ = preprocessing.Do_Task(img_base64, '', mock_request)
    times['OpenCV CUDA'] = (time.perf_counter() - start_time) * 1000
    results['OpenCV CUDA'] = decode_image_base64(result_cv_cuda_base64)
    print(f"耗時: {times['OpenCV CUDA']:.3f} ms")

    # 4. PyCUDA 版本
    print("\n--- 執行 PyCUDA 版本 ---")
    try:
        start_time = time.perf_counter()
        result_pycuda_base64, _ = preprocessing_cuda.Do_Task_CUDA(img_base64, '', mock_request)
        times['PyCUDA'] = (time.perf_counter() - start_time) * 1000
        results['PyCUDA'] = decode_image_base64(result_pycuda_base64)
        print(f"耗時: {times['PyCUDA']:.3f} ms")
    except Exception as e:
        print(f"PyCUDA 執行失敗: {e}")
        times['PyCUDA'] = -1
        results['PyCUDA'] = None

    # --- 比較結果 ---
    print("\n--- 結果比較 ---")
    if results.get('CPU') is not None and results.get('OpenCV CUDA') is not None:
        compare_images(results['CPU'], results['OpenCV CUDA'], "CPU", "OpenCV CUDA")
    
    if results.get('CPU') is not None and results.get('PyCUDA') is not None:
        compare_images(results['CPU'], results['PyCUDA'], "CPU", "PyCUDA")

    if results.get('OpenCV CUDA') is not None and results.get('PyCUDA') is not None:
        compare_images(results['OpenCV CUDA'], results['PyCUDA'], "OpenCV CUDA", "PyCUDA")

if __name__ == '__main__':
    # --- 設定要測試的任務與參數 ---
    
    # 測試 'Blur' (Mean Blur)
    run_benchmark(
        task_name='Blur',
        params={'blur_size': 15}
    )

    # 測試 'Guassian' (Gaussian Blur)
# 測試 'TransformationFlip'
    run_benchmark(
        task_name='TransformationFlip',
        params={'flip_type': 'XY'}
    )

    # 測試 'BilateralFilter'
    run_benchmark(
        task_name='BilateralFilter',
        params={'diameter': 15, 'sigmaColor': 80, 'sigmaSpace': 80}
    )

    # 測試 'Canny'
    run_benchmark(
        task_name='Canny',
        params={'threshold1': 3, 'threshold2': 60}
    )

    # 測試 'WarpAffine' (模擬一個輕微的縮放與平移)
    affine_matrix_str = '[1.1, 0.1, 50, -0.1, 1.1, 30]'
    run_benchmark(
        task_name='WarpAffine',
        params={'affine_matrix': affine_matrix_str}
    )
    run_benchmark(
        task_name='Guassian',
        params={'gaussian_blur_size': 15, 'gaussian_blur_sigma': 5}
    )

    # 測試 'GammaCorrection'
    run_benchmark(
        task_name='GammaCorrection',
        params={'lut_value': 1.8}
    )
    
    # 測試 'TransformationDegrees'
    run_benchmark(
        task_name='TransformationDegrees',
        params={'degree': '90'}
    )

    # 可以在此處加入更多測試案例...