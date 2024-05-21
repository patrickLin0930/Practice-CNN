#ipynb
# Download kaggle data in Google Colab
! pip install -q kaggle
from google.colab import files
files.upload()

! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

#kaggle 中的 copy api command
!kaggle datasets download -d drgfreeman/rockpaperscissors

#建立一個資料夾用以存放解壓縮後的檔案
! mkdir scissors-rock-paper 

# 解壓縮scissorsrockpaper到剛剛建立的資料夾
! unzip rockpaperscissors.zip -d scissors-rock-paper 

##2188張圖片(710、726、752)(300*200 pixel(RGB))
#(1)#先想分類剪刀石頭布適合用哪種ai
#(2)#把數據集分為訓練/測試集
#(3)#從小框架開始慢慢調整模型，用以避免over fitting

##圖像分類問題先使用cnn卷積神經網絡
## CNN 模型的最後一層添加一個 softmax 激活函數來輸出每個可能手勢的概率分佈，選擇概率最高的手勢作為模型的預測結果
##標籤1/2/3/4
##對應paper/rock/rps-cv-images/scissors