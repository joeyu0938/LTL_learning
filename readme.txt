HDR 圖片放在 Loader/images_HDR
LDR 圖片放在 Loader/images_LDR
3D regression numpy array 放在 tend_npy

#### LDR 圖片要先放在Loader/images_LDR
都只能在LTL_learning 資料夾下執行
主檔案 python inference.py
    -次檔案 python Loader/Filter_pytorch.py 將 HDR 轉換 成regression 
    -次檔案 python LDR2HDRinference.py  將 LDR 轉換成 HDR


testing image 是HDR to regression 的結果
testing code  是隨便的測試而已