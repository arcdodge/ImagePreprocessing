import os
import numpy as np
import cv2
import time
import base64
from unittest.mock import MagicMock
import argparse

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

def create_mock_request(task_name, params, acceleration="CPU"):
    """建立一個模擬的 Flask request 物件"""
    # getlist 應該為每個鍵返回一個值列表
    form_data = {
        'preprocess_task': [f'["{task_name}"]'],
        'acceleration': [acceleration]
    }
    for key, value in params.items():
        form_data[key] = [str(value)]
    
    request = MagicMock()
    request.form.getlist.side_effect = lambda k: form_data.get(k, [])
    
    def mock_get(key, default=None):
        value_list = form_data.get(key)
        if value_list:
            return value_list[0]
        return default
    request.form.get.side_effect = mock_get
    
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

def run_benchmark(task_name, params, acceleration="CPU", width=None, height=None, image_path=None):
    """執行指定任務的效能測試與比較"""
    # 1. 準備輸入資料
    img_to_use = None
    
    if image_path:
        img_to_use = cv2.imread(image_path)
        if img_to_use is None:
            print(f"錯誤: 無法讀取圖片 {image_path}")
            return None
        
        # 如果使用者從命令列指定了寬高，則縮放圖片
        if width is not None and height is not None:
            print(f"讀取指定圖片: {image_path}，並縮放至 {width}x{height}")
            img_to_use = cv2.resize(img_to_use, (width, height))
        else:
            print(f"讀取指定圖片: {image_path} (使用原始尺寸)")
    
    else:
        # 如果沒有指定圖片，則生成測試圖片
        # 如果使用者沒有指定寬高，使用預設值 500x500
        w = width if width is not None else 500
        h = height if height is not None else 500
        print(f"生成測試圖片 (尺寸: {w}x{h})")
        img_to_use = np.zeros((h, w, 3), dtype=np.uint8)
        img_to_use[:, :, 0] = np.tile(np.linspace(0, 255, w), (h, 1))
        img_to_use[:, :, 1] = np.tile(np.linspace(0, 255, h).reshape(h, 1), (1, w))
        img_to_use[:, :, 2] = 128

    final_h, final_w, _ = img_to_use.shape
    print(f"\n{'='*20} 測試任務: {task_name} (加速: {acceleration}, 最終尺寸: {final_w}x{final_h}) {'='*20}")
    print(f"參數: {params}")
    
    img_base64 = nparray_to_pngbase64str(img_to_use)
    mock_request = create_mock_request(task_name, params, acceleration)

    # --- 執行並計時 ---
    start_time = time.perf_counter()
    
    # 根據 acceleration 參數呼叫對應的處理函式
    if acceleration == "PyCUDA":
        print("\n--- 執行 PyCUDA 版本 ---")
        try:
            result_base64, _ = preprocessing_cuda.Do_Task_CUDA(img_base64, '', mock_request)
        except Exception as e:
            print(f"PyCUDA 執行失敗: {e}")
            result_base64 = None
    else: # "None" 或 "OpenCV CUDA"
        print(f"\n--- 執行 CPU/OpenCV 版本 (由 preprocessing 模組自動選擇) ---")
        if acceleration == "CVCUDA":
            preprocessing.CUDA_AVAILABLE = True
        else:
            preprocessing.CUDA_AVAILABLE = False
        # preprocessing 模組會根據 acceleration 參數和 CUDA_AVAILABLE 狀態自行決定
        result_base64, _ = preprocessing.Do_Task(img_base64, '', mock_request)

    elapsed_time = (time.perf_counter() - start_time) * 1000
    print(f"耗時: {elapsed_time:.3f} ms")

    # 解碼並回傳結果以供比較
    if result_base64:
        return decode_image_base64(result_base64)
    return None

    # --- 比較結果 ---
    print("\n--- 結果比較 ---")
    if results.get('CPU') is not None and results.get('OpenCV CUDA') is not None:
        compare_images(results['CPU'], results['OpenCV CUDA'], "CPU", "OpenCV CUDA")
    
    if results.get('CPU') is not None and results.get('PyCUDA') is not None:
        compare_images(results['CPU'], results['PyCUDA'], "CPU", "PyCUDA")

    if results.get('OpenCV CUDA') is not None and results.get('PyCUDA') is not None:
        compare_images(results['OpenCV CUDA'], results['PyCUDA'], "OpenCV CUDA", "PyCUDA")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="執行影像處理函式的效能測試")
    parser.add_argument('--width', type=int, default=None, help='測試影像的寬度 (若提供 image_path 則用於縮放)')
    parser.add_argument('--height', type=int, default=None, help='測試影像的高度 (若提供 image_path 則用於縮放)')
    parser.add_argument('--image_path', type=str, default=None, help='指定要測試的圖片路徑 (若不提供則生成預設圖片)')
    args = parser.parse_args()

    # --- 設定要測試的任務與參數 ---
    tasks_to_test = [
        {'name': 'Blur', 'params': {'blur_size': 15}},
        {'name': 'TransformationFlip', 'params': {'flip_type': 'XY'}},
        {'name': 'BilateralFilter', 'params': {'diameter': 15, 'sigmaColor': 80, 'sigmaSpace': 80}},
        {'name': 'Canny', 'params': {'threshold1': 3, 'threshold2': 60}},
        {'name': 'WarpAffine', 'params': {'affine_matrix': '[1.1, 0.1, 50, -0.1, 1.1, 30]'}},
        {'name': 'Guassian', 'params': {'gaussian_blur_size': 15, 'gaussian_blur_sigma': 5}},
        {'name': 'GammaCorrection', 'params': {'lut_value': 1.8}},
        {'name': 'TransformationDegrees', 'params': {'degree': '90'}},
    ]

    for task in tasks_to_test:
        results = {}
        # 執行三種加速模式
        results['CPU'] = run_benchmark(task_name=task['name'], params=task['params'], acceleration="CPU", width=args.width, height=args.height, image_path=args.image_path)
        results['OpenCV_CUDA'] = run_benchmark(task_name=task['name'], params=task['params'], acceleration="CVCUDA", width=args.width, height=args.height, image_path=args.image_path)
        results['PyCUDA'] = run_benchmark(task_name=task['name'], params=task['params'], acceleration="PyCUDA", width=args.width, height=args.height, image_path=args.image_path)
        
        # --- 結果比較 ---
        print("\n--- 結果比較 ---")
        if results.get('CPU') is not None and results.get('OpenCV_CUDA') is not None:
            compare_images(results['CPU'], results['OpenCV_CUDA'], "CPU", "OpenCV CUDA")
        
        if results.get('CPU') is not None and results.get('PyCUDA') is not None:
            compare_images(results['CPU'], results['PyCUDA'], "CPU", "PyCUDA")

        if results.get('OpenCV_CUDA') is not None and results.get('PyCUDA') is not None:
            compare_images(results['OpenCV_CUDA'], results['PyCUDA'], "OpenCV CUDA", "PyCUDA")