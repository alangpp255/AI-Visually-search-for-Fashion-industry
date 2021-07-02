# 自動化標籤與圖文搜索推薦

程式主要分為三個部分
AI 服裝標籤自動化: 運用fastai套件與resnet-34訓練標籤辨識model，可辨識服裝種類(category)和服裝樣式(pattern)
以圖搜圖: 可分析圖片顏色(HSL)並運用圖片灰階計算圖形相似度
文字搜尋:支援中英文和模糊比對，並新增服飾關鍵字庫

## Datasets

- 圖片資料集cvat_picture: 聯絡台大商業智慧實驗室獲得圖片資料，下載後放入同位置cvat_picture
- 其他資料集:https://drive.google.com/drive/folders/1aA9ztBVIcl9Mlr0__jYlzteSqrOozc7B?usp=sharing 下載後將資料集放入相同名稱目錄

## Models

- fastai_cat.pkl: 用於服裝種類(catagory)辨識
- fastai_pattern.pkl: 用於服裝樣式(pattern)辨識
- All the models trained with Resnet34 backbone

## 使用
事前準備: 
取得cvat_picture圖片資料集，下載後放入同位置cvat_picture
下載資料集，將其放入相同名稱目錄
1. AI 標籤自動化
- category_recog_training/fashion-category-recognition.ipynb 用於訓練category model
- pattern_recog_training/fashion-pattern-recognition.ipynb 用於訓練pattern model
- 使用tag_pred.py 操作model 進行自動化標籤

2. 以圖搜圖 v_search 
- Input: Image local path
- 使用V_search.py進行Input image的顏色分析與圖像相似度計算
- 最終依照顏色篩選結果與圖形相似度由近至遠排序，回傳資料庫圖片path

3. 文字搜索 s_search
- 使用方式：呼叫s_search函式，並依序傳入{欲搜尋之文字、文字與圖片檔對應之data frame、希望合併的搜尋結果}作為函式之參數
- 以此利用S_search.py 操作model 進行文字搜索
