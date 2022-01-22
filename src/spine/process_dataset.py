import json
import numpy as np
import os
import sys
import cv2
import base64
from PIL import Image
import io
from tqdm import tqdm

def process_json_to_img(base_dir, seed):

    json_path = [i for i in os.listdir(os.path.join(base_dir)) if os.path.splitext(os.path.basename(i))[1]=='.json'][seed]

    try:
        with open(os.path.join(base_dir, json_path)) as f:
            json_pic = json.load(f)
        img_data = json_pic['imageData']
            
        buf = io.BytesIO(base64.b64decode(str(img_data)))
        img = np.array(Image.open(buf))

        np.random.seed(0)
        num_col = len(json_pic['shapes'])
        colors = [(np.random.choice(range(256), size=3)).tolist() for i in range(num_col)]

        for idx, i in enumerate(json_pic['shapes']):
            pts = np.array(i['points'])
            img = cv2.polylines(img,np.int32([pts]), 1, colors[idx], 15)
        return img

    except Exception as e:
        print(name)
        print(e)
        return None
        
    

