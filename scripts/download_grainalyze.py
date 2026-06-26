"""
🌾 下载 Grainalyze 数据集并整理为 COCO 格式

Grainalyze - Roboflow Universe (CC BY 4.0)
https://universe.roboflow.com/rice-6hvoz/grainalyze

类别:
  - whole_grain      → 完整粒 (wanzheng)
  - broken_grain     → 破损粒 (posun)
  - chalky_grain     → 垩白粒/霉变粒 (meibian)
  - discolored_grain → 异色粒 (yise)

Usage:
    # 需要先在 https://app.roboflow.com/ 注册获取免费 API Key
    python scripts/download_grainalyze.py --api-key YOUR_ROBOFLOW_API_KEY

    或不带参数交互式输入:
    python scripts/download_grainalyze.py
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path


# ============================================================
# 类别映射: Grainalyze → 我们的命名
# ============================================================
CATEGORY_MAP = {
    'whole_grain':      {'id': 0, 'name': 'wanzheng', 'cn': '完整粒'},
    'broken_grain':     {'id': 1, 'name': 'posun',    'cn': '破损粒'},
    'chalky_grain':     {'id': 2, 'name': 'meibian',  'cn': '垩白粒'},
    'discolored_grain': {'id': 3, 'name': 'yise',     'cn': '异色粒'},
}


def parse_args():
    parser = argparse.ArgumentParser(description='下载 Grainalyze 数据集')
    parser.add_argument('--api-key', type=str, default=None,
                        help='Roboflow API Key (免费注册: https://app.roboflow.com/)')
    parser.add_argument('--workspace', type=str, default='rice-6hvoz',
                        help='Roboflow workspace')
    parser.add_argument('--project', type=str, default='grainalyze',
                        help='Roboflow project name')
    parser.add_argument('--version', type=int, default=None,
                        help='数据集版本号 (默认: 最新版)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录 (默认: 项目根目录)')
    parser.add_argument('--format', type=str, default='coco',
                        choices=['coco', 'coco-segmentation', 'yolov8'],
                        help='导出格式')
    return parser.parse_args()


def get_api_key(args_key):
    """获取 API Key (命令行参数 > 环境变量 > 交互输入)"""
    if args_key:
        return args_key
    env_key = os.environ.get('ROBOFLOW_API_KEY')
    if env_key:
        return env_key
    print("\n🔑 需要 Roboflow API Key (免费注册: https://app.roboflow.com/)")
    print("   注册后在 Settings → API Keys 中获取\n")
    return input("请输入 API Key: ").strip()


def download_dataset(api_key, workspace, project_name, version, output_dir, export_format):
    """从 Roboflow 下载数据集"""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("❌ 需要安装 roboflow: pip install roboflow")
        sys.exit(1)

    print(f"\n📥 连接 Roboflow...")
    rf = Roboflow(api_key=api_key)

    print(f"📋 获取项目: {workspace}/{project_name}")
    project = rf.workspace(workspace).project(project_name)

    # 获取版本
    if version:
        version_obj = project.version(version)
        print(f"📌 使用版本: v{version}")
    else:
        versions = project.versions()
        if not versions:
            print("❌ 没有找到可用版本")
            sys.exit(1)
        version_obj = versions[-1]  # 最新版
        print(f"📌 最新版本: v{version_obj.version}")

    print(f"   ID: {version_obj.id}")
    print(f"   创建时间: {version_obj.created}")

    # 下载
    print(f"\n⬇️  开始下载 (格式: {export_format})...")
    print(f"   目标目录: {output_dir}")

    dataset = version_obj.download(export_format, location=output_dir)

    print(f"✅ 下载完成: {dataset.location}")
    return dataset.location


def find_coco_json(download_dir):
    """查找下载的 COCO JSON 文件"""
    results = {'train': None, 'valid': None, 'test': None}

    for root, dirs, files in os.walk(download_dir):
        for f in files:
            if not f.endswith('.json'):
                continue
            fname_lower = f.lower()
            if '_train' in fname_lower:
                results['train'] = os.path.join(root, f)
            elif '_valid' in fname_lower:
                results['valid'] = os.path.join(root, f)
            elif '_test' in fname_lower:
                results['test'] = os.path.join(root, f)
            elif 'train' in root.lower() and 'annotations' in root.lower():
                results['train'] = os.path.join(root, f)
            elif ('valid' in root.lower() or 'val' in root.lower()) and 'annotations' in root.lower():
                results['valid'] = os.path.join(root, f)
            elif 'test' in root.lower() and 'annotations' in root.lower():
                results['test'] = os.path.join(root, f)

    # 如果上面没找到，尝试按名称
    for root, dirs, files in os.walk(download_dir):
        for f in files:
            if f.endswith('.json') and 'annotation' in f.lower():
                fpath = os.path.join(root, f)
                fpath_lower = fpath.lower()
                if 'train' in fpath_lower and not results['train']:
                    results['train'] = fpath
                elif ('valid' in fpath_lower or 'val' in fpath_lower) and not results['valid']:
                    results['valid'] = fpath
                elif 'test' in fpath_lower and not results['test']:
                    results['test'] = fpath

    return results


def convert_coco_annotations(src_json_path, dst_json_path, split_name):
    """转换 COCO 标注: 重映射类别 + 更新路径"""
    print(f"\n🔄 转换 {split_name} 标注...")

    with open(src_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. 重映射类别
    old_to_new_cat = {}
    orig_categories = data.get('categories', [])

    print(f"   原始类别: {[c['name'] for c in orig_categories]}")

    for cat in orig_categories:
        cat_name = cat['name'].lower().replace(' ', '_').replace('-', '_')
        if cat_name in CATEGORY_MAP:
            old_to_new_cat[cat['id']] = CATEGORY_MAP[cat_name]['id']
            print(f"   ✅ {cat['name']} → {CATEGORY_MAP[cat_name]['name']} ({CATEGORY_MAP[cat_name]['cn']})")
        else:
            print(f"   ⚠ 未知类别: {cat['name']} → 跳过")

    # 更新类别定义
    data['categories'] = [
        {"id": v['id'], "name": v['name'], "supercategory": "grain"}
        for v in CATEGORY_MAP.values()
    ]

    # 2. 更新图片路径
    for img in data.get('images', []):
        if 'file_name' in img:
            # 提取文件名
            fname = os.path.basename(img['file_name'].replace('\\', '/'))
            img['file_name'] = f"{split_name}\\{fname}"

    # 3. 更新标注的 category_id
    skipped = 0
    for ann in data.get('annotations', []):
        old_cid = ann['category_id']
        if old_cid in old_to_new_cat:
            ann['category_id'] = old_to_new_cat[old_cid]
            # 如果没有 bbox 但有 segmentation，从 segmentation 计算 bbox
            if 'bbox' not in ann or not ann['bbox']:
                if 'segmentation' in ann and ann['segmentation']:
                    seg = ann['segmentation']
                    if isinstance(seg, list) and len(seg) > 0:
                        if isinstance(seg[0], list):
                            points = seg[0]
                        else:
                            points = seg
                        if len(points) >= 4:
                            xs = points[0::2]
                            ys = points[1::2]
                            x_min, x_max = min(xs), max(xs)
                            y_min, y_max = min(ys), max(ys)
                            ann['bbox'] = [x_min, y_min, x_max - x_min, y_max - y_min]
                            ann['area'] = (x_max - x_min) * (y_max - y_min)
            elif 'area' not in ann or ann.get('area', 0) == 0:
                bbox = ann.get('bbox', [0, 0, 0, 0])
                ann['area'] = bbox[2] * bbox[3] if len(bbox) >= 4 else 0
        else:
            skipped += 1

    if skipped:
        print(f"   ⚠ 跳过 {skipped} 个未映射类别的标注")

    # 4. 按30%比例分出 val 和 test
    # Roboflow 通常只分 train/valid/test，如果只有 train/valid，我们保持原样
    # 实际上 Roboflow 下载的 COCO 已经是三分的

    # 保存
    os.makedirs(os.path.dirname(dst_json_path), exist_ok=True)
    with open(dst_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    n_imgs = len(data.get('images', []))
    n_anns = len(data.get('annotations', []))
    print(f"   ✅ 保存: {n_imgs} 图片, {n_anns} 标注 → {dst_json_path}")


def copy_images(download_dir, project_img_dir):
    """复制图片到项目目录"""
    print(f"\n📁 复制图片到: {project_img_dir}")

    # 查找图片文件
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    found_images = {'train': [], 'valid': [], 'test': []}

    for root, dirs, files in os.walk(download_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in img_extensions:
                continue
            fpath = os.path.join(root, f)
            fpath_lower = fpath.lower()
            if 'train' in fpath_lower:
                found_images['train'].append(fpath)
            elif 'valid' in fpath_lower or 'val' in fpath_lower:
                found_images['valid'].append(fpath)
            elif 'test' in fpath_lower:
                found_images['test'].append(fpath)

    # 如果通过路径没分出来，检查是否有单独的 images 目录结构
    if not any(found_images.values()):
        print("   尝试备选目录结构...")
        for split in ['train', 'valid', 'test']:
            for root, dirs, files in os.walk(os.path.join(download_dir, split)):
                for f in files:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in img_extensions:
                        found_images[split].append(os.path.join(root, f))

    # 复制
    split_name_map = {'train': 'train', 'valid': 'val', 'test': 'test'}
    total_copied = 0

    for src_split, dst_split in split_name_map.items():
        dst_dir = os.path.join(project_img_dir, dst_split)
        os.makedirs(dst_dir, exist_ok=True)
        for src_path in found_images.get(src_split, []):
            dst_path = os.path.join(dst_dir, os.path.basename(src_path))
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                total_copied += 1

    print(f"   复制了 {total_copied} 张图片")
    for split, imgs in found_images.items():
        print(f"     {split}: {len(imgs)} 张")


def main():
    args = parse_args()

    # 确定输出目录
    if args.output:
        project_dir = args.output
    else:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print('=' * 70)
    print('🌾 下载 Grainalyze 谷物品质检测数据集')
    print('=' * 70)
    print(f'   源: Roboflow Universe - rice-6hvoz/grainalyze')
    print(f'   License: CC BY 4.0')
    print(f'   目标目录: {project_dir}')
    print()

    # 获取 API Key
    api_key = get_api_key(args.api_key)
    if not api_key:
        print("❌ 需要 API Key 才能下载")
        sys.exit(1)

    # 临时下载目录
    temp_dir = os.path.join(project_dir, '.temp_download')
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # 1. 下载数据集
        download_location = download_dataset(
            api_key=api_key,
            workspace=args.workspace,
            project_name=args.project,
            version=args.version,
            output_dir=temp_dir,
            export_format=args.format,
        )

        # 2. 查找 COCO JSON 文件
        print(f"\n🔍 查找标注文件...")
        coco_files = find_coco_json(download_location)
        for split, path in coco_files.items():
            status = '✅' if path else '❌'
            print(f"   {status} {split}: {path or '未找到'}")

        # 3. 转换标注
        anno_dir = os.path.join(project_dir, 'date', 'coco', 'annotations')
        split_map = {'train': 'instances_train.json', 'valid': 'instances_val.json', 'test': 'instances_test.json'}

        for src_split, dst_filename in split_map.items():
            if coco_files.get(src_split):
                convert_coco_annotations(
                    coco_files[src_split],
                    os.path.join(anno_dir, dst_filename),
                    'val' if src_split == 'valid' else src_split,
                )

        # 4. 复制图片
        img_dir = os.path.join(project_dir, 'date', 'coco', 'images')
        copy_images(download_location, img_dir)

        # 5. 清理临时文件
        print(f"\n🧹 清理临时文件...")
        shutil.rmtree(temp_dir, ignore_errors=True)

        # 6. 统计
        print('\n' + '=' * 70)
        print('📊 最终数据集统计:')
        total_img = 0
        total_ann = 0
        for split_key, filename in split_map.items():
            json_path = os.path.join(anno_dir, filename)
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    d = json.load(f)
                n_img = len(d.get('images', []))
                n_ann = len(d.get('annotations', []))
                total_img += n_img
                total_ann += n_ann
                print(f'   {split_key}: {n_img} 张图片, {n_ann} 个标注')
        print(f'   总计: {total_img} 张图片, {total_ann} 个标注')
        print(f'   类别: wanzheng(完整粒), posun(破损粒), meibian(垩白粒), yise(异色粒)')
        print()
        print('✅ 全部完成！')
        print()
        print('下一步:')
        print('  python scripts/check_dataset.py')
        print('  python tools/train.py configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py')

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        # 清理
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
