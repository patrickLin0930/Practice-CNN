# libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
# 畫出layer圖片
from tensorflow.keras.utils import plot_model
# 讀取數據集並劃分訓練集和測試集
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models

print(tf.__version__)


# 定義圖片大小
img_width, img_height = 150, 100
# 設置數據集目錄
dataset_dir = "scissors-rock-paper"

# 讀取圖片和標籤
def load_images_and_labels(dataset_dir):
    images = []
    labels = []
    class_names = sorted(os.listdir(dataset_dir))
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                # 跳過文件跟非圖片檔案
                if os.path.isdir(file_path) or not file_path.endswith(('.png', '.jpg', '.jpeg')):
                    continue
                # 讀取圖片數據(包含標籤)跟調整大小
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (img_width, img_height))
                images.append(image)
                labels.append(label-1)
    return np.array(images), np.array(labels)

# load圖片跟標籤
images, labels = load_images_and_labels(dataset_dir)

# 打亂數據集
shuffle_indices = np.random.permutation(len(images))
images = images[shuffle_indices]
labels = labels[shuffle_indices]

# 把數據集合切分成訓練/測試集合
from sklearn.model_selection import train_test_split
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_images = train_images / 255.0
test_images = test_images / 255.0

# 轉換為 NumPy 數組
images = np.array(images)
labels = np.array(labels)





#檢查訓練集中的第一張影像
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

##定義模型
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(100, 150, 3)),  
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),  
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),  
    layers.MaxPooling2D((2, 2)),
    ##進入全連接層之前要先將輸入都展平(flatten)
    layers.Flatten(),
    ##連接層
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='sigmoid')
])

##配置模型(優化器、損失函數、評估標準)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
##損失函數選Sparse Categorical Crossentropy，損失值範圍在0~ln(n)#n為類別數


##輸出模型的summary
model.summary()
##繪製網路圖片
plot_model(model, to_file='model_plot.png', show_shapes=True)

##訓練模型 重複跑10次
model.fit(train_images, train_labels, epochs=8)
#print("Minimum label:", np.min(labels))
#print("Maximum label:", np.max(labels))

##評估準確性，比較模型在測試資料集上的表現
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
##當機器學習模型在新的、以前未見過的輸入上的表現比在訓練資料上的表現更差時，就會發生過度擬合
print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)
# 保存模型
#model.save('trained_model.h5')
##----------------------------------------
# load 模型
#from tensorflow.keras.models import load_model
#loaded_model = load_model('trained_model.h5')

# 對新數據預測
#new_images = ...  # 新數據
#predictions = loaded_model.predict(new_images)
##----------------------------------------






#----------------------------------------------------------------
##以此配置模型(優化器、損失函數、評估標準) 用以比對數據
#model.compile(optimizer='adam',
              #loss='sparse_categorical_crossentropy',
              #metrics=['accuracy']) 
##20240517跑8次 loss: 0.0354 - accuracy: 0.9886
#Test accuracy: 0.9497717022895813
#結論:over fitting
#model = models.Sequential([
    #layers.Conv2D(16, (3, 3), activation='relu', input_shape=(100, 150, 3)),  
    #layers.MaxPooling2D((2, 2)),
    #layers.Conv2D(32, (3, 3), activation='relu'),  
    #layers.MaxPooling2D((2, 2)),
    #layers.Conv2D(64, (3, 3), activation='relu'),  
    #layers.MaxPooling2D((2, 2)),
    #layers.Flatten(),
    #layers.Dense(64, activation='relu'),
    #layers.Dense(4, activation='sigmoid')
#])
#----------------------------------------------------------------
#----------------------------------------------------------------
#20240517跑7次 loss: 0.0674 - accuracy: 0.9817
##Test accuracy: 0.9543378949165344
#結論:over fitting
#model = models.Sequential([
    #layers.Conv2D(16, (3, 3), activation='relu', input_shape=(100, 150, 3)),  
    #layers.MaxPooling2D((2, 2)),
    #layers.Conv2D(32, (3, 3), activation='relu'),  
    #layers.MaxPooling2D((2, 2)),
    #layers.Flatten(),
    #layers.Dense(32, activation='relu'),
    #layers.Dense(4, activation='softmax')
#])
#----------------------------------------------------------------
#loss: 0.0886 - accuracy: 0.9697
#Test accuracy: 0.9748858213424683
#結論:無over fitting
#model = models.Sequential([
    #layers.Conv2D(16, (3, 3), activation='relu', input_shape=(100, 150, 3)),  
    #layers.MaxPooling2D((2, 2)),
    #layers.Conv2D(32, (3, 3), activation='relu'),  
    #layers.MaxPooling2D((2, 2)),
    #layers.Conv2D(64, (3, 3), activation='relu'), 
    #layers.MaxPooling2D((2, 2)),
    ##進入全連接層之前要先將輸入都展平(flatten)
    #layers.Flatten(),
    ##連接層
    #layers.Dense(64, activation='relu'),
    #layers.Dropout(0.5),
    #layers.Dense(4, activation='softmax')
#])
#----------------------------------------------------------------

