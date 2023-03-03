HDR 圖片放在 Loader/images_HDR
LDR 圖片放在 Loader/images_LDR
3D regression numpy array 放在 tend_npy

都只能在LTL_learning 執行
主檔案 python inference.py
    -次檔案 python Loader/Filter_pytorch.py
    -次檔案 python LDR2HDRinference.py


testing folder 是測試而已