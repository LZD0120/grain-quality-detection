"""
数据集完整性检查脚本

Usage:
    python scripts/check_dataset.py
"""

import json
import os
import sys
from pathlib import Path


def check_coco_json(json_path: str, img_dir: str) -> dict:
    """检查 COCO JSON 格式标注文件"""
    issues = []
    stats = {}

    if not os.path.exists(json_path):
        return {'error': f'文件不存在: {json_path}'}, [f'文件不存在: {json_path}']

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 基本信息
    stats['images'] = len(data.get('images', []))
    stats['annotations'] = len(data.get('annotations', []))
    stats['categories'] = len(data.get('categories', []))

    # 检查类别定义
    categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    print(f"  类别定义: {categories}")
    print(f"  图片数量: {stats['images']}")
    print(f"  标注数量: {stats['annotations']}")

    # 检查每张图片是否存在
    missing_images = []
    for img in data.get('images', []):
        img_path = os.path.join(img_dir, img['file_name'])
        if not os.path.exists(img_path):
            missing_images.append(img['file_name'])

    if missing_images:
        issues.append(f'缺失 {len(missing_images)} 张图片')
        for fn in missing_images[:5]:
            issues.append(f'  - {fn}')

    # 检查标注的 category_id 是否有效
    valid_cat_ids = set(categories.keys())
    for ann in data.get('annotations', []):
        if ann['category_id'] not in valid_cat_ids:
            issues.append(f"标注 id={ann['id']} 的 category_id={ann['category_id']} 无效")

    # 检查是否有空标注的图片
    image_ids_with_ann = set(ann['image_id'] for ann in data.get('annotations', []))
    image_ids_all = set(img['id'] for img in data.get('images', []))
    empty_images = image_ids_all - image_ids_with_ann
    if empty_images:
        issues.append(f'{len(empty_images)} 张图片没有任何标注')

    stats['missing_images'] = len(missing_images)
    stats['empty_images'] = len(empty_images)
    stats['issues'] = len(issues)

    return stats, issues


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    anno_dir = os.path.join(base_dir, 'date', 'coco', 'annotations')
    img_dir = os.path.join(base_dir, 'date', 'coco', 'images')

    print('=' * 60)
    print('谷物颗粒品质分级检测 - 数据集检查')
    print('=' * 60)

    all_ok = True
    for split in ['train', 'val', 'test']:
        print(f'\n--- 检查 {split} 集 ---')
        json_path = os.path.join(anno_dir, f'instances_{split}.json')
        img_split_dir = os.path.join(img_dir, split)

        stats, issues = check_coco_json(json_path, img_split_dir)

        if issues:
            all_ok = False
            for issue in issues:
                print(f'  ⚠ {issue}')
        else:
            print(f'  ✅ 检查通过')

    # 总体统计
    print('\n' + '=' * 60)
    print('数据集总体统计:')
    for split in ['train', 'val', 'test']:
        json_path = os.path.join(anno_dir, f'instances_{split}.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            n_imgs = len(data.get('images', []))
            n_anns = len(data.get('annotations', []))
            print(f'  {split}: {n_imgs} 张图片, {n_anns} 个标注')

    if all_ok:
        print('\n✅ 所有检查通过！')
    else:
        print('\n⚠ 存在问题，请根据提示修复')


if __name__ == '__main__':
    main()
