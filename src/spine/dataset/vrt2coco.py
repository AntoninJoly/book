import json
import numpy as np
import os
import sys
import cv2
import base64
from PIL import Image
import io
from tqdm import tqdm

def process_json_to_img(base_dir):
    list_json = os.listdir(os.path.join(base_dir,'json'))
    
    for json_path in tqdm(list_json, position=0):
        name, ext = os.path.splitext(os.path.basename(json_path))
        try:
            with open(os.path.join(base_dir, 'json', json_path)) as f:
                json_pic = json.load(f)
            img_data = json_pic['imageData']
            
            buf = io.BytesIO(base64.b64decode(str(img_data)))
            img = Image.open(buf)

            img.save(os.path.join(base_dir, 'img/%s.png'%(name)))
            img = np.array(img)

            np.random.seed(0)
            num_col = len(json_pic['shapes'])
            colors = [(np.random.choice(range(256), size=3)).tolist() for i in range(num_col)]

            for idx, i in enumerate(json_pic['shapes']):
                pts = np.array(i['points'])
                img = cv2.polylines(img,np.int32([pts]),1,colors[idx], 8)
            cv2.imwrite(os.path.join(base_dir, 'vis/%s.png'%(name)), img)
        except Exception as e:
            print(name)
            print(e)
    return [os.path.join(base_dir, 'vis', i) for i in os.listdir(os.path.join(base_dir, 'vis'))]

