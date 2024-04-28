# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import json
import time
import logging
import tools.infer.utility as utility
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
from ppocr.utils.utility import get_image_file_list, check_and_read
from ppocr.utils.logging import get_logger
from tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop

logger = get_logger()

_labels = []


class TextSystem(object):
    def __init__(self, args):
        if not args.show_log:
            logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.drop_score = args.drop_score
        self.args = args
        self.crop_image_res_index = 0
        self.num_classes = args.num_classes
        self.magnify_pixel = int(args.magnify_pixel)

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno + self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def magnify_label_boxes_pixel(self, boxes):
        if self.args.det_box_type == "quad" and self.magnify_pixel != 0:
            result_boxes = []
            index = 0
            for box in boxes:
                if index % 4 == 0:
                    box[0] = box[0] - self.magnify_pixel
                    box[1] = box[1] - self.magnify_pixel
                elif index % 4 == 1:
                    box[0] = box[0] + self.magnify_pixel
                    box[1] = box[1] - self.magnify_pixel
                elif index % 4 == 2:
                    box[0] = box[0] + self.magnify_pixel
                    box[1] = box[1] + self.magnify_pixel
                elif index % 4 == 3:
                    box[0] = box[0] - self.magnify_pixel
                    box[1] = box[1] + self.magnify_pixel
                index += 1
                result_boxes.append(box)
            result_boxes = np.array(result_boxes)
            return result_boxes
        else:
            return boxes

    def __call__(self, img, cls=True):
        if img is None:
            logger.debug("no valid image provided")
            return None, None

        ori_im = img.copy()
        dt_boxes, elapse, classes = self.text_detector(img)
        if len(classes) != self.num_classes:
            return None, None

        tmp_dt_boxes = []
        index = 0
        for box in dt_boxes:
            label_name = ''
            if len(_labels) > 0:
                label_name = _labels[classes[index]]

            tmp_dt_boxes.append({
                'points': self.magnify_label_boxes_pixel(box),
                'label': label_name
            })
            index += 1
        dt_boxes = np.array(tmp_dt_boxes)

        if dt_boxes is None:
            logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            return None, None
        else:
            logger.debug("dt_boxes num : {}, elapsed : {}".format(
                len(dt_boxes), elapse))
        img_crop_list = []

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno]['points'])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.debug("rec_res num  : {}, elapsed : {}".format(
            len(rec_res), elapse))
        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
                                   rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result[0], rec_result[1]
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        return filter_boxes, filter_rec_res


def main(args):
    image_file_list = get_image_file_list(args.image_dir)
    image_file_list = image_file_list[args.process_id::args.total_process_num]
    text_sys = TextSystem(args)
    draw_img_save_dir = args.draw_img_save_dir
    os.makedirs(draw_img_save_dir, exist_ok=True)
    save_results = []

    logger.info(
        "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320', "
        "if you are using recognition model with PP-OCRv2 or an older version, please set --rec_image_shape='3,32,320"
    )

    total_time = 0
    _st = time.time()
    for idx, image_file in enumerate(image_file_list):

        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            img = cv2.imread(image_file)
        if not flag_pdf:
            if img is None:
                logger.debug("error in loading image:{}".format(image_file))
                continue
            imgs = [img]
        else:
            page_num = args.page_num
            if page_num > len(img) or page_num == 0:
                page_num = len(img)
            imgs = img[:page_num]
        for index, img in enumerate(imgs):
            starttime = time.time()
            dt_boxes, rec_res = text_sys(img)
            if dt_boxes is None or rec_res is None:
                continue
            elapse = time.time() - starttime
            total_time += elapse
            if len(imgs) > 1:
                logger.debug(
                    str(idx) + '_' + str(index) + "  Predict time of %s: %.3fs"
                    % (image_file, elapse))
            else:
                logger.debug(
                    str(idx) + "  Predict time of %s: %.3fs" % (image_file,
                                                                elapse))
            for text, score in rec_res:
                logger.debug("{}, {:.3f}".format(text, score))

            res = [{
                "transcription": rec_res[i][0],
                "points": np.array(dt_boxes[i]['points']).astype(np.int32).tolist(),
                'difficult': False,
                'key_cls': dt_boxes[i]['label']
            } for i in range(len(dt_boxes))]
            save_pred = image_file + "\t" + json.dumps(
                res, ensure_ascii=False) + "\n"
            save_results.append(save_pred)

    logger.info("The predict total time is {}".format(time.time() - _st))

    with open(
            os.path.join(draw_img_save_dir, "Label.txt"),
            'w',
            encoding='utf-8') as f:
        f.writelines(save_results)


if __name__ == "__main__":
    args = utility.parse_args()
    label_list = args.label_list_path
    labels = []
    if label_list is not None:
        if isinstance(label_list, str) and os.path.isfile(args.label_list_path):
            with open(label_list, "r+", encoding="utf-8") as f:
                for line in f.readlines():
                    labels.append(line.replace("\n", ""))
        else:
            labels = label_list
    _labels = labels

    main(args)
