# 自動化標籤與圖文搜索推薦

程式主要分為三個部分
AI 服裝標籤自動化: 運用fastai套件與resnet-34訓練標籤辨識model，可辨識服裝種類(category)和服裝樣式(pattern)
以圖搜圖: 可分析圖片顏色(HSL)並運用圖片灰階計算圖形相似度
文字搜尋:支援中英文和模糊比對，並新增服飾關鍵字庫

## Datasets

- 圖片資料集cvat_picture: 聯絡台大商業智慧實驗室獲得圖片資料，下載後放入同位置cvat_picture
- recog_traning 相關資料集: 存放在model_train_data，此資料集為尚未進行自動化標籤之原始資料集
- clothing_tags_complete(HSL).csv: 存放在tag_data，此資料集為已完成自動化標籤之資料集，v_search.py 和 s_search.py 使用 


## Models

- fastai_cat.pkl: 用於服裝種類(catagory)辨識
- fastai_pattern.pkl: 用於服裝樣式(pattern)辨識
- All the models trained with Resnet34 backbone


## 使用
1. 取得cvat_picture圖片資料集，下載後放入同位置cvat_picture
category_recog_training/fashion-category-recognition.ipynb 用於訓練category
pattern_recog_training/fashion-pattern-recognition.ipynb 用於訓練pattern

2. v_search 
3. s_search
