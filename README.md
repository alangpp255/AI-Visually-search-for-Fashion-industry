# Automated tag and image search recommendations

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

## Usage使用
Preparation: 
Get the cvat_picture image data set, download it and put it in the same location as cvat_picture
Download the dataset, put it into a directory of the same name
1. AI clothing label automation
- category_recog_training/fashion-category-recognition.ipynb : to train category model
- pattern_recog_training/fashion-pattern-recognition.ipynb : to train pattern model
- Use tag_pred.py to operate the model for automated tagging

2. Visual Search Model v_search 
- Input: Image local path
- Use V_search.py for Input image color analysis and image similarity calculation
- Finally, sort the results from near to far according to the similarity between the color filtering results and the graphics, and return the database image path.
- (The above process can be realise more clearly using the ipynb in Color detection and similarity ranking with perceptual hash algorithm)

3. Text search s_search
- How to use: Call the s_search function, and sequentially pass in {the text to be searched, the data frame corresponding to the text and image files, the search results you want to combine} as the parameters of the function
- In this way, use S_search.py to operate the model for text search
