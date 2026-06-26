"""
Label Studio 导出 JSON → COCO 格式转换脚本

Label Studio 导出的 JSON 包含所有标注在一个文件中，
此脚本将其按 train/val/test 划分并转换为标准 COCO 格式。

Usage:
    python scripts/convert_labelstudio.py --input output/project-xxx.json --output date/coco/annotations/
"""

import json
import os
import random
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description='Label Studio JSON → COCO 转换')
    parser.add_argument('--input', required=True, help='Label Studio 导出的 JSON 文件')
    parser.add_argument('--output', default='date/coco/annotations/', help='输出目录')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='训练集比例')
    parser.add_argument('--val-ratio', type=float, default=0.15, help='验证集比例')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--categories', nargs='+', default=['wanzheng', 'posun', 'meibian', 'yise'],
                        help='类别名称列表 (按 Label Studio 中的顺序)')
    return parser.parse_args()


def convert_labelstudio_to_coco(ls_json_path: str, output_dir: str,
                                 train_ratio: float = 0.7, val_ratio: float = 0.15,
                                 seed: int = 42, category_names: list = None):
    """
    将 Label Studio 的导出 JSON 转换为 COCO 格式
    """
    if category_names is None:
        category_names = ['wanzheng', 'posun', 'meibian', 'yise']

    # 加载 Label Studio JSON
    with open(ls_json_path, 'r', encoding='utf-8') as f:
        ls_data = json.load(f)

    # 构建 COCO 基础结构
    coco_base = {
        "info": {
            "year": 2026,
            "version": "1.0",
            "description": "谷物颗粒品质分级检测数据集",
            "contributor": "Label Studio",
            "url": "",
            "date_created": "2026-06-22 00:00:00.000000"
        },
        "licenses": [
            {"id": 1, "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}
        ],
        "categories": [
            {"id": i, "name": name, "supercategory": "grain"}
            for i, name in enumerate(category_names)
        ]
    }

    # 解析标注任务
    tasks = ls_data if isinstance(ls_data, list) else [ls_data]

    all_images = []
    all_annotations = []
    image_id = 0
    annotation_id = 0

    for task in tasks:
        if 'data' not in task or 'image' not in task['data']:
            continue

        # 图像信息
        img_path = task['data']['image']
        img_name = os.path.basename(img_path)

        # 查找实际图片大小
        width = 1920  # 默认值
        height = 1080

        annotations = task.get('annotations', [])
        if not annotations:
            continue

        # 取第一个标注结果
        for ann in annotations:
            result = ann.get('result', [])
            if not result:
                continue

            img_entry = {
                "width": width,
                "height": height,
                "id": image_id,
                "file_name": img_name
            }
            all_images.append(img_entry)

            for r in result:
                if r.get('type') != 'rectanglelabels':
                    continue

                label_name = r['value'].get('rectanglelabels', [None])[0]
                if label_name not in category_names:
                    continue

                category_id = category_names.index(label_name)

                # 转换坐标: Label Studio (x, y, width, height) 百分比 → COCO 绝对 bbox
                x_pct = r['value']['x'] / 100.0
                y_pct = r['value']['y'] / 100.0
                w_pct = r['value']['width'] / 100.0
                h_pct = r['value']['height'] / 100.0

                bbox_x = x_pct * width
                bbox_y = y_pct * height
                bbox_w = w_pct * width
                bbox_h = h_pct * height

                ann_entry = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [],
                    "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                    "ignore": 0,
                    "iscrowd": 0,
                    "area": bbox_w * bbox_h
                }
                all_annotations.append(ann_entry)
                annotation_id += 1

            image_id += 1
            break  # 每个 task 只取第一个标注结果

    # 按比例随机划分
    random.seed(seed)
    indices = list(range(len(all_images)))
    random.shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        'train': indices[:n_train],
        'val': indices[n_train:n_train + n_val],
        'test': indices[n_train + n_val:]
    }

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 为每个划分生成 COCO JSON
    for split_name, split_indices in splits.items():
        split_images = [all_images[i] for i in split_indices]
        split_image_ids = {img['id'] for img in split_images}

        # 重新映射 image_id (从0开始)
        old_to_new_id = {old_id: new_id for new_id, old_id in enumerate(
            sorted([img['id'] for img in split_images]))}

        for img in split_images:
            new_id = old_to_new_id[img['id']]
            # 更新文件路径
            img['file_name'] = f"{split_name}\\{img['file_name'].split('\\\\')[-1].split('/')[-1]}"
            img['id'] = new_id

        split_annotations = []
        new_ann_id = 0
        for ann in all_annotations:
            if ann['image_id'] in split_image_ids:
                ann['id'] = new_ann_id
                ann['image_id'] = old_to_new_id[ann['image_id']]
                split_annotations.append(ann)
                new_ann_id += 1

        coco_split = coco_base.copy()
        coco_split['images'] = split_images
        coco_split['annotations'] = split_annotations

        output_path = os.path.join(output_dir, f'instances_{split_name}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_split, f, ensure_ascii=False, indent=2)

        print(f"  ✅ {split_name}: {len(split_images)} 图片, {len(split_annotations)} 标注 → {output_path}")

    print(f"\n总计: {n_total} 图片, {len(all_annotations)} 标注")
    print(f"划分: train={n_train}, val={n_val}, test={n_total - n_train - n_val}")


if __name__ == '__main__':
    args = parse_args()
    convert_labelstudio_to_coco(
        args.input, args.output,
        args.train_ratio, args.val_ratio,
        args.seed, args.categories
    )
