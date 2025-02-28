# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import argparse
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import time
import torchvision.transforms as T
from models import build_model
# from models3_1 import build_model

import warnings
warnings.filterwarnings("ignore")

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    parser.add_argument('--img_path', type=str, default='./demo/1.jpg',
                        help="Path of the test image")
    parser.add_argument('--topk', default=10, type=int)

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=150, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=150, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    # parser.add_argument('--resume', default='./ckpt/checkpoint_best.pth', help='resume from checkpoint')
    parser.add_argument('--resume', default='./ckpt/checkpoint_best.pth', help='resume from checkpoint')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--rel_loss_coef_so', default=4, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # distributed training parameters
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser


# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def AinB(a, B):
    for i in range(len(B)):
        b = B[i]
        if (a == b).all():
            return False, i
    return True, len(B)

def AinB2(a, b):
    if np.linalg.norm(a - b) < 100:
        return True
    return False


def FilterRepeat(keep, probas, probas_sub, probas_obj, sub_bboxes, obj_bboxes):
    pro = torch.argmax(probas[keep], dim=-1)
    pro_sub = torch.argmax(probas_sub[keep], dim=-1)
    pro_obj = torch.argmax(probas_obj[keep], dim=-1)
    temp = torch.stack((pro, pro_sub, pro_obj), 0).numpy().T

    sub_boxes = sub_bboxes.detach().numpy()
    obj_boxes = obj_bboxes.detach().numpy()

    # print(temp)
    L = len(pro)
    kp = [True]

    for idx in range(1, L):
        d, idx_ = AinB(temp[idx, :], temp[:idx, :])
        if d:
            kp.append(True)
        elif AinB2(sub_boxes[idx, :], sub_boxes[idx_, :]) and AinB2(obj_boxes[idx, :], obj_boxes[idx_, :]):
            kp.append(False)
        else:
            kp.append(True)

    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    x = keep_queries[torch.tensor(kp)]
    T = torch.zeros(len(keep), dtype=torch.bool)
    T[x] = True
    return T

transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def function(model, im):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    with torch.no_grad():
        start = time.time()
        # propagate through the model
        # for _ in range(10):
        outputs = model(img)
        end = time.time()
        print("处理时间：{}".format(end - start))
        # keep only predictions with 0.+ confidence
        probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
        probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
        probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
        keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                                probas_obj.max(-1).values > 0.3))

        # convert boxes from [0; 1] to image scales
        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)
        # 去除重复的结果，如果不需要注释下面3行
        keep = FilterRepeat(keep, probas, probas_sub, probas_obj, sub_bboxes_scaled, obj_bboxes_scaled)
        sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
        obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

        topk = args.topk
        keep_queries = torch.nonzero(keep, as_tuple=True)[0]

        # cannot perform reduction function max on tensor with no elements because the operation does not have an identity ?
        indices = torch.argsort(
            -probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[
                  :topk]
        # print(indices)
        keep_queries = keep_queries[indices]

        # use lists to store the outputs via up-values
        conv_features, dec_attn_weights_sub, dec_attn_weights_obj, dec_attn_weights_ety = [], [], [], []

        hooks = [
            model.backbone[-2].register_forward_hook(
                lambda self, input, output: conv_features.append(output)
            ),
            model.transformer.decoder.layers[-1].cross_attn_entity.register_forward_hook(
                lambda self, input, output: dec_attn_weights_ety.append(output[1]))
        ]

        with torch.no_grad():
            # propagate through the model
            outputs = model(img)

            for hook in hooks:
                hook.remove()

            # don't need the list anymore
            conv_features = conv_features[0]
            dec_attn_weights_sub, dec_attn_weights_obj = torch.split(dec_attn_weights_ety[0], 1, dim=0)

            # get the feature map shape
            h, w = conv_features['0'].tensors.shape[-2:]
            im_w, im_h = im.size

            Res = []
            fig, axs = plt.subplots(ncols=len(indices), nrows=3, figsize=(22, 7))
            fig2 = plt.figure(figsize=(22, 14))
            ax2 = fig2.add_subplot(111)
            ax2.imshow(im)
            for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                    zip(keep_queries, axs.T, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
                ax = ax_i[0]
                # idx_1 = dec_attn_weights_sub_max[0, idx]
                ax.imshow(dec_attn_weights_sub[0, idx].view(h, w))
                ax.axis('off')
                ax.set_title(f'sub id: {idx.item()}' + ' ' + CLASSES[probas_sub[idx].argmax()], fontsize=18)

                ax = ax_i[1]
                # idx_2 = dec_attn_weights_obj_max[0, idx]
                ax.imshow(dec_attn_weights_obj[0, idx].view(h, w))
                ax.axis('off')
                ax.set_title(f'obj id: {idx.item()}' + ' ' + CLASSES[probas_obj[idx].argmax()], fontsize=18)

                ax = ax_i[2]
                ax.imshow(im)
                ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                           fill=False, color='blue', linewidth=2.5))
                ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                           fill=False, color='orange', linewidth=2.5))

                ax2.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                            fill=False, color='blue', linewidth=2.0))
                ax2.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                            fill=False, color='blue', linewidth=2.0))
                ax2.add_line(plt.Line2D((int(0.5 * sxmin + 0.5 * sxmax), int(0.5 * oxmin + 0.5 * oxmax)),
                                        (int(0.5 * symin + 0.5 * symax), int(0.5 * oymin + 0.5 * oymax)),
                                        color='red', linewidth=4))
                ax2._add_text(plt.text(int(0.5 * sxmin + 0.5 * sxmax), int(0.5 * symin + 0.5 * symax),
                                       CLASSES[probas_sub[idx].argmax()], color='orange', fontsize=45))
                ax2._add_text(plt.text(int(0.5 * oxmin + 0.5 * oxmax), int(0.5 * oymin + 0.5 * oymax),
                                       CLASSES[probas_obj[idx].argmax()], color='orange', fontsize=45))

                ax.axis('off')
                ax.set_title(CLASSES[probas_sub[idx].argmax()] + ' ' + REL_CLASSES[probas[idx].argmax()] + ' ' + CLASSES[
                    probas_obj[idx].argmax()], fontsize=18)
                Res.append([CLASSES[probas_sub[idx].argmax()], REL_CLASSES[probas[idx].argmax()], CLASSES[
                    probas_obj[idx].argmax()]])
        # del outputs
    return fig, fig2, Res


def main(args):
    args.topk = 20
    model, _, _ = build_model(args)
    ckpt = torch.load(args.resume)
    model.load_state_dict(ckpt['model'])
    model.eval()
    with torch.no_grad():
        # img_path = args.img_path
        img_path = args.img_path
        save_img_path = './demo/5_res_light/'
        # 创建输出文件夹
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)

        if os.path.isdir(img_path):
            filelist = os.listdir(img_path)
            num = 0
            # sum = 0
            for file in filelist:
                try:
                    print(file)
                    if os.path.exists(os.path.join(save_img_path, file.split('.j')[0] + '.txt')):
                        print("{}已处理\n".format(file))
                        continue
                    im = Image.open(os.path.join(img_path, file))
                    fig, fig2, Res = function(model, im)
                    fig2.savefig(os.path.join(save_img_path, file))
                    f = open(os.path.join(save_img_path, file.split('.j')[0] + '.txt'), 'w+')
                    for d in Res:
                        f.write("{}\t{}\t{}\n".format(d[0], d[1], d[2]))
                    f.close()
                    print("检测到{}个三元组".format(len(Res)))
                    num += 1
                    if num > 50:
                        return
                    # sum +=
                    # print(sum / num)
                    plt.cla()
                    plt.clf()
                    fig.close()
                    fig2.close()
                except:
                    pass

            #
        else:
            im = Image.open(img_path)
            fig, fig2, _ = function(model, im)

            fig.tight_layout()
            fig2.tight_layout()
            plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()

    # VG classes
    CLASSES = ['N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
               'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
               'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
               'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
               'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
               'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
               'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
               'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
               'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
               'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
               'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
               'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
               'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
               'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

    REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                   'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                   'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                   'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                   'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                   'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

    main(args)
