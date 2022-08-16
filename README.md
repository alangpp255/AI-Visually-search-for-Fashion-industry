# 自動化標籤與圖文搜索推薦

程式主要分為三個部分
AI 服裝標籤自動化: 運用fastai套件與resnet-34訓練標籤辨識model，可辨識服裝種類(category)和服裝樣式(pattern)
以圖搜圖: 可分析圖片顏色(HSL)並運用圖片灰階計算圖形相似度
文字搜尋:支援中英文和模糊比對，並新增服飾關鍵字庫

The program is mainly divided into three parts 
AI clothing label automation:  Based on fastai kit and resnet-34 label recognition model, it can identify clothing category (category) and clothing style (pattern) 
Visual Search Model: Analyzed image color (HSL) and use image grayscale to calculate image similarity
Text search: support Chinese and English and fuzzy matching, and add clothing keyword database

## Datasets

- cvat_picture: Contact the National Taiwan University Business Intelligence Laboratory to obtain the picture information, download it and put it in the same location cvat_picture
- other datasets:https://drive.google.com/drive/folders/1aA9ztBVIcl9Mlr0__jYlzteSqrOozc7B?usp=sharing 

## Models

- fastai_cat.pkl: For clothing category (catagory) identification
- fastai_pattern.pkl: Used for clothing pattern recognition
- All the models trained with Resnet34 backbone

## 使用
事前準備: 
取得cvat_picture圖片資料集，下載後放入同位置cvat_picture
下載資料集，將其放入相同名稱目錄
1. AI clothing label automation
- category_recog_training/fashion-category-recognition.ipynb 用於訓練category model
- pattern_recog_training/fashion-pattern-recognition.ipynb 用於訓練pattern model
- 使用tag_pred.py 操作model 進行自動化標籤

2. Visual Search Model v_search 
- Input: Image local path
- 使用V_search.py進行Input image的顏色分析與圖像相似度計算
- 最終依照顏色篩選結果與圖形相似度由近至遠排序，回傳資料庫圖片path
- (以上可使用Color detection and similarity ranking with perceptual hash algorithm 中的ipynb操作更清楚)

3. Text search s_search
- 使用方式：呼叫s_search函式，並依序傳入{欲搜尋之文字、文字與圖片檔對應之data frame、希望合併的搜尋結果}作為函式之參數
- 以此利用S_search.py 操作model 進行文字搜索
