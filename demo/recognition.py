import matplotlib.pyplot as plt
from PIL import Image
import os
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

def get_config():
    config = Cfg.load_config_from_name('vgg_transformer')
    config['pretrained'] = '/content/ArtistText/load_models/model_final.pth'
    config['weights'] = '/content/ArtistText/load_models/model_final.pth'
    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch']=True
    return config

config = get_config()
detector = Predictor(config)

def convert_to_polygon(bbox):
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2, y1, x2, y2, x1, y2]

def predict(img_path, bboxs):
    results = []
    img = Image.open(img_path)
    for bbox in bboxs:
        cropped_img = img.crop(bbox)
        pred = detector.predict(cropped_img)
        result = {'text': pred, 'bbox': convert_to_polygon(bbox)}
        results.append(result)
    
    return results

def make_submissions(img_path, results, output_dir='./submissions'):
    img_name = os.path.basename(img_path)
    img_name = os.path.splitext(img_name)[0]
    # format: x1, y1, x2, y2, x3, y3, x4, y4, text
    with open(os.path.join(output_dir, img_name + '.txt'), 'w') as f:
        for result in results:
            f.write(','.join([str(x) for x in result['bbox']]) + ',' + result['text'] + '\n')