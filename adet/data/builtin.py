import os

from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from pathlib import Path
from .datasets.text import register_text_instances

# register plane reconstruction

_PREDEFINED_SPLITS_PIC = {
    "pic_person_train": ("pic/image/train", "pic/annotations/train_person.json"),
    "pic_person_val": ("pic/image/val", "pic/annotations/val_person.json"),
}

metadata_pic = {
    "thing_classes": ["person"]
}
data_root = Path('YOUR_ROOT_IMG_PATH')
token_json_dir = Path('YOUR_ROOT_ANNOTATION_PATH')

_PREDEFINED_SPLITS_TEXT = {
    # datasets with bezier annotations
    "totaltext_train": (os.path.join(data_root,"totaltext/train_images"),os.path.join(token_json_dir,"totaltext_train_96voc_bart_base.json")),
    "totaltext_val": (os.path.join(data_root,"totaltext/test_images"), os.path.join(token_json_dir,"totaltext_test.json")),
    "syntext1_train": (os.path.join(data_root,"syntext1/images/syntext_word_eng"), os.path.join(token_json_dir,"syntext1_train_96voc_bart_base.json")),
    "syntext2_train": (os.path.join(data_root,"syntext2/images/emcs_imgs"), os.path.join(token_json_dir,"syntext2_train_96voc_bart_base.json")),
    "icdar2013_train": (os.path.join(data_root,"icdar2013/train_images"), os.path.join(token_json_dir,"icdar13_train_96voc_bart_base.json")),
    "icdar2015_train": (os.path.join(data_root,"icdar2015/train_images"), os.path.join(token_json_dir,"icdar15_train_96voc_bart_base.json")),
    "ic15_test": (os.path.join(data_root,"icdar2015/test_images"), os.path.join(token_json_dir,"icdar15_test.json")),
    "mltbezier_word_train": (os.path.join(data_root,"mlt2017/MLT_train_images"), os.path.join(token_json_dir,"mlt2017_train_bart_base.json")),
    "textocr_train": (os.path.join(data_root,"textocr/train_images"), os.path.join(token_json_dir,"text_ocr_1_bart_base.json")),
    "ctw1500_word_poly_train": (os.path.join(data_root,"ctw1500/ctw1500/train_images"), os.path.join(token_json_dir,"ctw_train_bart_base.json")),
    "ctw1500_word_poly_test": (os.path.join(data_root,"ctw1500/ctw1500/test_images"), os.path.join(token_json_dir,"ctw_test_bart_base.json")),

}

metadata_text = {
    "thing_classes": ["text"]
}


def register_all_coco(root="datasets", voc_size_cfg=194, num_pts_cfg=25):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PIC.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata_pic,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TEXT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_text_instances(
            key,
            metadata_text,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            voc_size_cfg,
            num_pts_cfg
        )


register_all_coco()
