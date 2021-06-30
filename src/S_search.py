# -*- coding:utf-8 -*-

from fuzzy_match import algorithims, match
from fuzzywuzzy import fuzz, process

import jieba
# from textgo import Preprocess
import argparse
import logging
import re
import csv
import pandas as pd
import cv2
import numpy as np

en = '[\u0041-\u005a|\u0061-\u007a|-|\s]+' #regular expression to find english words
zh = '[\u4e00-\u9fa5]+'					   #regular expression to find chinese words

jieba.setLogLevel(logging.INFO)

#Create dictionary about tags with given terminology
tags = {
'Coat': 'Coat', '外套': 'Coat', 'Culottes': 'Culottes', '褲裙': 'Culottes', 
'Dress': 'Dress', '洋裝': 'Dress', 'Henley': 'Henley', '亨利衫': 'Henley', 'Hoodie': 'Hoodie or Sweatshirt', '連帽衫': 'Hoodie or Sweatshirt', '帽T': 'Hoodie or Sweatshirt',
'Jacket': 'Jacket', '夾克': 'Jacket', 'Jeans': 'Jeans', '牛仔褲': 'Jeans', 'Jersey': 'Jersey', '球衣': 'Jersey', 'Jumpsuit': 'Jumpsuit', 
'連身褲': 'Jumpsuit', 'Kaftan': 'Kaftan', '長袍': 'Kaftan', 'Kimono': 'Kimono', '和服': 'Kimono', 'Leggings': 'Legging', '束褲': 'Legging',
'Onesie': 'Onesie', '連身衣': 'Onesie', 'Poncho': 'Poncho', '雨衣': 'Poncho', 'Romper': 'Romper', '短連身衣': 'Romper',
'Shorts': 'Shorts', '短褲': 'Shorts', 'Skirt': 'Skirt', '短裙': 'Skirt', 'Sweater': 'Sweater', '毛衣': 'Sweater',
'Pants': 'Pant', '長褲': 'Pant', 'Tank': 'Tank', '無袖上衣': 'Tank', '無袖': 'Tank', 'Tee': 'Short Sleeve Top', 'Tshirt': 'Short Sleeve Top', 'Short Sleeve Top':'Short Sleeve Top', '短袖上衣':'Short Sleeve Top', '短袖':'Short Sleeve Top',
'Trunks': 'Trunks', '泳褲': 'Trunks', 'Blazer': 'Blazer', '西裝外套': 'Blazer', 'Shirt': 'Shirt', '襯衫': 'Shirt',
'Long Sleeve Top': 'Long Sleeve Top', '長袖上衣': 'Long Sleeve Top','長袖': 'Long Sleeve Top',
'Capri': 'Capri', '七分褲': 'Capri', 'Cardigan': 'Cardigan', '開襟衫': 'Cardigan', 'Chinos': 'Chinos', '卡其褲': 'Chinos',
'Diving suit': 'Diving suit', '潛水裝': 'Diving suit', 'Sports Bra': 'Sports Bra', '運動胸罩': 'Sports Bra', 'Polo': 'Polo', 'Polo衫': 'Polo',
'Swimwear': 'Swimwear', '游泳衣': 'Swimwear', 'Vest': 'Vest', '背心': 'Vest', 'Sleepwear': 'Sleepwear', '睡衣': 'Sleepwear',
'純色': 'Solid', 'Solid': 'Solid', '條紋': 'Strips', 'Strips': 'Strips', '花': 'Floral', 'Floral': 'Floral', 
'圖案': 'Placement', 'Placement': 'Placement', '麻花紗': 'Melange', 'Melange': 'Melange', '熱帶': 'Tropical', 'Tropical': 'Tropical',
 '動物印花': 'Animal print', 'Animal print': 'Animal print', 'IKAT':'IKAT', '斑點': 'Spots', 'Spots': 'Spots', '抽象': 'Abstract', 'Abstract': 'Abstract', 
 '漸層': 'Gradient', 'Gradient': 'Gradient', '自然風景': 'Scenery', 'Scenery': 'Scenery', '幾何': 'Geometric', 'Geometric': 'Geometric', '格子': 'Plaids',
 'Plaids': 'Plaids', '佩斯里': 'Paisley', 'Paisley': 'Paisley', '迷彩': 'Camouflage', 'Camouflage': 'Camouflage', 'Tight':'Tight', '緊身褲':'Tight',
 'Red':'Red', '紅':'Red', 'Orange':'Orange', '橘色':'Orange', 'Yellow':'Yellow', '黃':'Yellow', 'Spring Green':'Spring Green', 'Green':'Green', 
 'Turquoise':'Turquoise', 'Cyan':'Cyan', 'Ocean':'Ocean', 'Blue':'Blue', 'Violet':'Violet', 'Magenta':'Magenta', 'Raspberry':'Raspberry' ,
 '春綠':'Spring Green', '綠':'Green', '綠松':'Turquoise', '青色':'Cyan', '海藍':'Ocean', '水藍':'Ocean', '藍':'Blue', '藍紫':'Violet', '洋紅':'Magenta', '紫紅':'Magenta', '樹莓':'Raspberry',
 'Dark Red':'Dark Red', '深紅':'Red', '暗紅':'Red', 'Dark Orange':'Dark Orange', '深橘色':'Dark Orange', 'Dark Yellow':'Dark Yellow', '深黃':'Dark Yellow', '暗黃':'Dark Yellow', 'Dark Spring Green':'Dark Spring Green', 'Dark Green':'Dark Green', 
 'Dark Turquoise':'Dark Turquoise', 'Dark Cyan':'Dark Cyan', 'Dark Ocean':'Dark Ocean', 'Dark Blue':'Dark Blue', 'Dark Violet':'Dark Violet', 'Dark Magenta':'Dark Magenta', 'Dark Raspberry':'Dark Raspberry' ,
 '深春綠':'Dark Spring Green', '深綠':'Dark Green', '深綠松':'Dark Turquoise', '深青色':'Dark Cyan', '深海藍':'Dark Ocean', '深水藍':'Dark Ocean', '深藍':'Dark Blue', '深藍紫':'Dark Violet', '深洋紅':'Dark Magenta', '深紫紅':'Dark Magenta', '深樹莓':'Dark Raspberry',
 'Light Red':'Light Red', '淺紅':'Light Red', '淡紅':'Light Red','Light Orange':'Light Orange', '淡橘色':'Light Orange', 'Light Yellow':'Light Yellow', '淺黃':'Light Yellow', '淡黃':'Light Yellow', 'Light Spring Green':'Light Spring Green', 'Light Green':'Light Green', 
 'Light Turquoise':'Light Turquoise', 'Light Cyan':'Light Cyan', 'Light Ocean':'Light Ocean', 'Light Blue':'Light Blue', 'Light Violet':'Light Violet', 'Light Magenta':'Light Magenta', 'Light Raspberry':'Light Raspberry' ,
 '淺春綠':'Light Spring Green', '淺綠':'Light Green', '淺綠松':'Light Turquoise', '淡青色':'Light Cyan', '淡海藍':'Light Ocean', '淡水藍':'Light Ocean', '淺藍':'Light Blue', '淡藍':'Light Blue', '淺藍紫':'Light Violet', '淺洋紅':'Light Magenta', '淡紫紅':'Light Magenta', '淺樹莓':'Light Raspberry',
 'Black':'Black', 'White':'White', 'Grey':'Grey', '黑':'Black', '白':'White', '灰':'Grey'
}

#addition dictionary for faster and more accurate searching
cat_list=["Capri","Completer","Diving suit","Dress","Hoodie or Sweatshirt","Jacket","Jeans","Jumpsuit"
,"Legging","Long Sleeve Top","Pant","Polo","Shirt","Short Sleeve Top","Shorts","Skirt","Sleepwear","Sports Bra"
,"Sweater","Swimwear","Tank","Tight","Vest"]



keys = [k for k in tags.keys()]


def search(text):
	chinese = re.findall(zh, text) #find all chinese words
	english = re.findall(en, text) #find all english words
	#print("English = ", english)
	str_zh = ''
	for ch in chinese: #concatenate found chinese word
		str_zh += ch
	
	words = jieba.cut(str_zh, cut_all = False) #cut the words using jieba
	# print(type(words))
	res = []
	
	#cut the chinese words if not in predefined keys with cutoff score=60
	if str_zh not in keys:
		
		for word in words:
			# print(word)
			same=0
			cal = process.extractBests(word, keys, score_cutoff = 60)
			print(process.extractBests(word, keys, score_cutoff = 60,limit=10))
			for i in res:
				if i[0] in [x[0] for x in cal]:
					same=1
					break
			if same	==0:	
				res.extend(cal)
	else:
		res.extend([(str_zh,100)])
			
		
	#cut the english words using fuzzy matching with a different cutoff score
	for word in english:
		res.extend(process.extractBests(word, keys, score_cutoff = 90, scorer=fuzz.token_set_ratio))
		print(process.extractBests(word, keys, score_cutoff = 70, scorer=fuzz.token_set_ratio, limit=10))
		
	
	res = [tags[x[0]] for x in res]
	return set(res)


def s_search(text, fc_df, cat):
	ret = search(text)  #start to search 
 
	print("Your expected tags: ", ret)
	ret = list(ret)

	#重複控制
	#for repeatedly searching
	if cat != '0':
		for x in ret:
			if x in cat_list:
				ret.remove(x)
				ret.append(cat)	
	 
	
	
	#assign result with weight to get search priority
	fc_df['score'] = fc_df['clothing'].isin(ret).astype(int)*3  + fc_df['pattern_0'].isin(ret).astype(int)*2 + fc_df['pattern_1'].isin(ret).astype(int) + fc_df['pattern_2'].isin(ret).astype(int) + fc_df['pattern_3'].isin(ret).astype(int)
	fc_df['score'] += fc_df['main_color'].isin(ret).astype(int) * 2.5 + fc_df['sub_color.1'].isin(ret).astype(int) + fc_df['sub_color.2'].isin(ret).astype(int)
	fc_df = fc_df.sort_values(by=['score'], ascending=False) #sort result accoirding to given score
	
	show = []
	pre, cnt = 0, 0
	for idx, data in fc_df.iterrows():
		cnt += 1
		
		if data['score'] > 0 and (cnt <= 5 or (data['score'] == pre and cnt <= 80)):
			image = '/'.join(('cvat_pictures', data['user'], data['task_name'], data['filename']))
			show.append(image)
			pre = data['score']
		else:
			break
	

	return show, ret


# 迷彩 外套
# 格子襯衫 sport bra
# 裙子 褲子 Diving Suit
# 外套 sport bra jeans
