# load 模型
from tensorflow.keras.models import load_model as lm
loaded_model = lm('trained_model.h5')
import numpy as np
import os
import cv2  # 或者使用其他影像處理庫，如Pillow

# 存儲影像數據和對應標籤的列表
new_images = []
labels = []

# 設置 test 資料夾的路徑
test_folder = "test"
# 遍歷 test 資料夾中的每個影像檔案
for filename in os.listdir(test_folder):
    # 構建影像檔案的完整路徑
    img_path = os.path.join(test_folder, filename)

    # 使用 OpenCV 加載影像
    # 如果您選擇使用 Pillow，請相應地更改此處的代碼
    image = cv2.imread(img_path)

    # 對影像進行預處理（如調整大小、歸一化等）
    # 此處省略預處理步驟，根據模型輸入的要求進行相應處理

    # 將預處理後的影像添加到影像列表中
    new_images.append(image)

    # 如果您有標籤檔案，您可以在此處添加代碼來加載影像標籤
    # 根據您的模型需求，如果您的模型不需要標籤，則可以忽略此步驟

# 將影像列表轉換為 NumPy 陣列（如果需要）
new_images = np.array(new_images)

# 現在，您可以使用加載的新影像數據進行預測

# 對新數據進行預測
new_images = ...  # 新數據
predictions = loaded_model.predict(new_images)
