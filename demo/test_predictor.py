from __future__ import division
import argparse
import os
from mmcv import Config
from mmcv.runner import load_checkpoint
from tqdm import tqdm
from mmfashion.core import AttrPredictor
from mmfashion.models import build_predictor
from mmfashion.utils import get_img_tensor


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMFashion Attribute Prediction Demo')
    parser.add_argument(
        '--input',
        type=str,
        help='input image path',
        default='/content/drive/My Drive/dataset/nordstrom/img/1')
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='checkpoint file',
        default='/content/mmfashion/checkpoint/Predict/vgg/global/latest.pth')
    parser.add_argument(
        '--config',
        help='test config file path',
        default='/content/mmfashion/configs/attribute_predict/global_predictor_vgg_attr.py')
    parser.add_argument(
        '--use_cuda', type=bool, default=True, help='use gpu or not')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    cfg.model.pretrained = None
    model = build_predictor(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.use_cuda:
        model.cuda()
    model.eval()
    attr_predictor = AttrPredictor(cfg.data.test)
    files=os.listdir(args.input)
    for i in tqdm(range(len(files))) :
      filename=files[i]
      #print(filename)      
      filename=args.input+'/'+filename
      img_tensor = get_img_tensor(filename, args.use_cuda) 
      attr_prob = model(img_tensor, attr=None, landmark=None, return_loss=False)
      attr_predictor.show_prediction(attr_prob,filename)


if __name__ == '__main__':
    main()
