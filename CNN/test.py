# 載入必要的模組
from google.colab.patches import cv2_imshow

# 載入模型
from tensorflow.keras.models import load_model as lm
import numpy as np
import os
import cv2  # 影像處理庫
from tensorflow.keras.preprocessing import image

# 載入訓練好的模型
loaded_model = lm('my_model.keras')

# 檢查模型的結構，了解每層的輸入/輸出形狀
loaded_model.summary()

# 存儲影像數據和對應標籤的列表
new_images = []
labels = []
file_names = []

# 設置 test 資料夾的路徑
test_folder = "test"

# 設定模型所需的影像大小（根據模型期望輸入大小為 100x150）
target_size = (100, 150)

# 遍歷 test 資料夾中的每個 PNG 影像檔案
for filename in os.listdir(test_folder):
    if filename.endswith(".png"):  # 只處理 PNG 檔案
        # 構建影像檔案的完整路徑
        img_path = os.path.join(test_folder, filename)

        # 使用 OpenCV 加載 PNG 影像
        image_data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # 確保影像載入成功
        if image_data is not None:
            # 調整影像大小到模型的輸入要求 (100x150)
            image_resized = cv2.resize(image_data, target_size)

            # 如果影像有 alpha 通道 (PNG 可能有4個通道)，轉換為 RGB（3個通道）
            if image_resized.shape[2] == 4:
                image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGRA2BGR)

            # 將影像數據轉換為 float32，並進行歸一化 (0-255 範圍轉為 0-1)
            image_resized = image_resized.astype('float32') / 255.0

            # 將影像添加到影像列表中
            new_images.append(image_resized)
            file_names.append(filename)  # 儲存檔案名稱以供顯示

        else:
            print(f"無法載入影像: {filename}")

# 將影像列表轉換為 NumPy 陣列
new_images = np.array(new_images)

# 使用模型對新影像數據進行預測（保持原來的三維影像形狀）
predictions = loaded_model.predict(new_images)

# 打印預測結果
predicted_classes = np.argmax(predictions, axis=1)

# 遍歷每張影像並顯示對應的預測結果
for idx, prediction in enumerate(predicted_classes):
    print(f"影像: {file_names[idx]}, 預測類別: {prediction}")
    
    # 將歸一化的影像轉換回原始範圍，便於顯示
    image_to_display = (new_images[idx] * 255).astype(np.uint8)

    # 顯示影像
    cv2_imshow(image_to_display)
