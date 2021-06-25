from fastai.vision.all import *
import fastai
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score

def get_x(r):
    return "C:/visual_search_project/static/img/cvat_pictures/"+r['image_name'] 
def get_y(r):
    return r['labels'].split(';')



class LabelSmoothingBCEWithLogitsLossFlat(BCEWithLogitsLossFlat):
    def __init__(self, eps:float=0.1, **kwargs):
        self.eps = eps
        super().__init__(thresh=0.2, **kwargs)
    
    def __call__(self, inp, targ, **kwargs):
        targ_smooth = targ.float() * (1. - self.eps) + 0.5 * self.eps
        return super().__call__(inp, targ_smooth, **kwargs)
    
    def __repr__(self):
        return "FlattenedLoss of LabelSmoothingBCEWithLogits()"

def pat_pred():
    
    metrics=[FBetaMulti(2.0, 0.2, average='samples'), partial(accuracy_multi, thresh=0.2)]
    wd      = 5e-7 #weight decay parameter
    opt_func = partial(ranger, wd=wd)
    learner = load_learner("C:/visual_search_project/myapp/model/fastai_pattern.pkl")
    
    return(learner.predict('C:/visual_search_project/image/image_crop.jpeg')[0])

def cat_pred():
    learner2 = load_learner("C:/visual_search_project/myapp/model/fastai_cat.pkl")
    return(learner2.predict('C:/visual_search_project/image/image_crop.jpeg')[0])
