"""
合成谷物颗粒图像生成器
生成500张逼真的合成谷物颗粒图像 + COCO格式标注

每张图像1920×1080，黑色绒布背景，5-15粒随机分布的谷物颗粒
4个类别各有明显视觉差异，用于训练目标检测模型

Usage:
    python scripts/generate_synthetic_data.py
"""

import os
import sys
import json
import random
import math
import argparse
import io

# 强制 UTF-8 输出 (解决 Windows GBK 编码问题)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFilter, ImageChops
except ImportError:
    print("请先安装 Pillow: pip install Pillow")
    sys.exit(1)


# ============================================================
# 配置参数
# ============================================================
CONFIG = {
    'img_width': 1920,
    'img_height': 1080,
    'grains_per_image': (5, 15),       # 每图颗粒数范围
    'grain_size': (90, 180),           # 颗粒长轴范围 (像素)
    'grain_aspect': (0.35, 0.55),      # 颗粒长短轴比
    'rotation_range': (0, 360),        # 旋转角度
    'margin': 40,                      # 边缘留白
    'overlap_threshold': 0.3,          # 最大允许重叠率

    # 数据集划分
    'n_train': 350,
    'n_val': 75,
    'n_test': 75,

    # 输出路径
    'output_dir': None,  # 将在 main() 中设置
    'seed': 42,
}


# ============================================================
# 类别定义
# ============================================================
CATEGORIES = [
    {
        'id': 0,
        'name': 'wanzheng',
        'name_cn': '完整粒',
        # 正常金黄色调
        'color_base': [(180, 140, 80), (210, 165, 100), (195, 150, 90), (200, 155, 85)],
        'texture': 'smooth',
        'has_crack': False,
        'has_mold': False,
        'color_abnormal': False,
    },
    {
        'id': 1,
        'name': 'posun',
        'name_cn': '破损粒',
        # 破损处颜色偏白/偏浅，整体仍为金色但有不规则边缘
        'color_base': [(175, 135, 75), (200, 150, 90), (190, 140, 80)],
        'texture': 'broken',
        'has_crack': True,
        'has_mold': False,
        'color_abnormal': False,
    },
    {
        'id': 2,
        'name': 'meibian',
        'name_cn': '霉变粒',
        # 灰绿色霉斑覆盖
        'color_base': [(120, 110, 70), (140, 120, 80), (100, 95, 65), (130, 105, 60)],
        'texture': 'moldy',
        'has_crack': False,
        'has_mold': True,
        'color_abnormal': False,
    },
    {
        'id': 3,
        'name': 'yise',
        'name_cn': '异色粒',
        # 异常颜色：偏白、偏深褐、偏黄
        'color_base': [(230, 200, 140), (80, 50, 30), (220, 190, 50), (240, 220, 160), (90, 55, 25)],
        'texture': 'discolored',
        'has_crack': False,
        'has_mold': False,
        'color_abnormal': True,
    },
]


# ============================================================
# 辅助函数
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def make_noise_texture(width, height, scale=4):
    """生成 Perlin-like 噪声纹理"""
    small_w, small_h = width // scale, height // scale
    noise = np.random.rand(small_h, small_w) * 255
    noise_img = Image.fromarray(noise.astype(np.uint8))
    noise_img = noise_img.resize((width, height), Image.BILINEAR)
    return np.array(noise_img, dtype=np.float32)


def draw_grain_base(draw, cx, cy, major_axis, minor_axis, angle_deg, color):
    """
    绘制基础谷物颗粒 (旋转椭圆)
    返回边界框 (x1, y1, x2, y2)
    """
    # 创建临时画布
    grain_img = Image.new('RGBA', (major_axis * 3, major_axis * 3), (0, 0, 0, 0))
    grain_draw = ImageDraw.Draw(grain_img)

    # 在中心绘制椭圆
    c = major_axis * 1.5
    bbox = [
        c - major_axis,
        c - minor_axis,
        c + major_axis,
        c + minor_axis,
    ]
    grain_draw.ellipse(bbox, fill=color)

    # 旋转
    grain_img = grain_img.rotate(angle_deg, resample=Image.BILINEAR, center=(c, c))

    # 裁剪到实际内容
    grain_img = grain_img.crop(grain_img.getbbox())
    if grain_img.size[0] == 0 or grain_img.size[1] == 0:
        return None

    # 计算实际边界
    gw, gh = grain_img.size
    px = int(cx - gw / 2)
    py = int(cy - gh / 2)

    return {
        'image': grain_img,
        'x': px,
        'y': py,
        'width': gw,
        'height': gh,
        'bbox': [px, py, gw, gh],
    }


def add_crack_effect(grain_info):
    """给破损粒添加裂纹效果"""
    img = grain_info['image'].copy()
    w, h = img.size
    arr = np.array(img, dtype=np.float32)

    # 裂纹: 一条不规则锯齿线
    crack_points = []
    cx, cy = w / 2, h / 2
    n_pts = random.randint(5, 12)
    for i in range(n_pts):
        t = i / (n_pts - 1)
        px = random.uniform(0.15, 0.85) * w
        py = random.uniform(0.2, 0.8) * h
        crack_points.append((px, py))

    # 绘制裂纹 (深色锯齿线 + 亮色边缘)
    crack_img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    crack_draw = ImageDraw.Draw(crack_img)

    if len(crack_points) >= 2:
        # 主裂纹 (深色)
        for width_val in [5, 3, 2]:
            color = (30, 20, 10, random.randint(150, 220)) if width_val > 3 else (80, 60, 40, random.randint(180, 255))
            pts_shifted = [(x + random.uniform(-2, 2), y + random.uniform(-2, 2)) for x, y in crack_points]
            crack_draw.line(pts_shifted, fill=color, width=width_val, joint='curve')

    # 合成
    result = Image.alpha_composite(img.convert('RGBA'), crack_img)

    grain_info['image'] = result
    return grain_info


def add_mold_effect(grain_info):
    """给霉变粒添加霉斑效果"""
    img = grain_info['image'].copy()
    w, h = img.size
    arr = np.array(img.convert('RGBA'), dtype=np.float32)

    # 在颗粒表面添加圆形霉斑
    n_spots = random.randint(6, 20)
    for _ in range(n_spots):
        sx = random.randint(int(w * 0.1), int(w * 0.9))
        sy = random.randint(int(h * 0.1), int(h * 0.9))
        sr = random.randint(4, max(5, min(w, h) // 6))

        # 霉斑颜色: 灰绿到深灰
        mold_color = (
            random.randint(60, 140),
            random.randint(70, 130),
            random.randint(55, 110),
            random.randint(100, 200),
        )

        # 在霉斑区域应用
        y_min = max(0, sy - sr)
        y_max = min(h, sy + sr)
        x_min = max(0, sx - sr)
        x_max = min(w, sx + sr)

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                dist = math.sqrt((x - sx) ** 2 + (y - sy) ** 2)
                if dist <= sr and arr[y, x, 3] > 10:
                    alpha_ratio = (1 - dist / sr) * 0.7
                    arr[y, x, 0] = arr[y, x, 0] * (1 - alpha_ratio) + mold_color[0] * alpha_ratio
                    arr[y, x, 1] = arr[y, x, 1] * (1 - alpha_ratio) + mold_color[1] * alpha_ratio
                    arr[y, x, 2] = arr[y, x, 2] * (1 - alpha_ratio) + mold_color[2] * alpha_ratio

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    grain_info['image'] = Image.fromarray(arr, 'RGBA')
    return grain_info


def add_shadow(grain_info):
    """给颗粒添加阴影效果 (投影)"""
    img = grain_info['image']
    w, h = img.size

    # 创建阴影
    shadow_offset = (3, 4)
    shadow = Image.new('RGBA', (w + 8, h + 8), (0, 0, 0, 0))
    shadow.paste(
        Image.new('RGBA', (w, h), (0, 0, 0, 80)),
        (4 + shadow_offset[0], 4 + shadow_offset[1])
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=3))

    # 先画阴影再画颗粒
    result = Image.new('RGBA', (w + 8, h + 8), (0, 0, 0, 0))
    result.paste(shadow, (0, 0))
    result.paste(img, (4, 4), img)

    grain_info['image'] = result
    grain_info['x'] -= 4
    grain_info['y'] -= 4
    grain_info['width'] += 8
    grain_info['height'] += 8
    grain_info['bbox'] = [
        grain_info['x'],
        grain_info['y'],
        grain_info['width'],
        grain_info['height'],
    ]
    return grain_info


def create_background(width, height):
    """创建黑色绒布背景 (带微纹理)"""
    # 基础暗色
    bg = np.ones((height, width, 3), dtype=np.uint8) * random.randint(18, 28)

    # 添加微弱噪声模拟绒布纹理
    noise = np.random.randint(0, 12, (height, width, 3), dtype=np.uint8)
    bg = np.clip(bg.astype(np.int16) + noise.astype(np.int16) - 5, 0, 255).astype(np.uint8)

    # 中心微亮渐变 - 使用像素级操作代替 Image.blend
    bg_img = Image.fromarray(bg)

    # 创建光照蒙版
    cx, cy = width / 2, height / 2
    max_dist = math.sqrt(cx**2 + cy**2)
    for y_step in range(0, height, 4):  # 每隔4像素采样加速
        for x_step in range(0, width, 4):
            dist = math.sqrt((x_step - cx)**2 + (y_step - cy)**2)
            factor = 1.0 - (dist / max_dist) * 0.15
            factor = max(0.92, factor)
            if factor > 1.0:
                continue
            # 对4x4块应用
            y_end = min(y_step + 4, height)
            x_end = min(x_step + 4, width)
            block = bg[y_step:y_end, x_step:x_end].astype(np.float32)
            block = block * factor + np.array([8, 7, 5], dtype=np.float32) * (1 - factor)
            bg[y_step:y_end, x_step:x_end] = np.clip(block, 0, 255).astype(np.uint8)

    return Image.fromarray(bg)


def check_overlap(new_bbox, existing_bboxes, img_w, img_h):
    """检查新边界框与已有框的重叠率"""
    x1, y1, w1, h1 = new_bbox
    area1 = w1 * h1
    if area1 <= 0:
        return True  # 无效框

    for eb in existing_bboxes:
        x2, y2, w2, h2 = eb
        # 计算交集
        ix1 = max(x1, x2)
        iy1 = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        inter_area = (ix2 - ix1) * (iy2 - iy1)
        area2 = w2 * h2
        overlap = inter_area / min(area1, area2)
        if overlap > CONFIG['overlap_threshold']:
            return True
    return False


def get_grain_color(category):
    """根据类别获取颗粒颜色 (随机微调)"""
    base = random.choice(category['color_base'])
    # 添加随机微调
    r = min(255, max(0, base[0] + random.randint(-12, 12)))
    g = min(255, max(0, base[1] + random.randint(-12, 12)))
    b = min(255, max(0, base[2] + random.randint(-12, 12)))
    return (r, g, b)


def generate_single_grain(category, img_w, img_h, existing_bboxes, max_attempts=30):
    """在图片上放置一个谷物颗粒"""
    cat_info = category

    for _ in range(max_attempts):
        # 随机长轴和短轴
        major = random.randint(*CONFIG['grain_size'])
        minor = int(major * random.uniform(*CONFIG['grain_aspect']))
        angle = random.randint(*CONFIG['rotation_range'])

        # 计算放置区域 (考虑旋转后的边界)
        rad = math.radians(angle)
        cos_a = abs(math.cos(rad))
        sin_a = abs(math.sin(rad))
        rot_w = major * cos_a + minor * sin_a
        rot_h = major * sin_a + minor * cos_a
        pad_w = int(rot_w * 1.3) + 20  # 留出阴影空间
        pad_h = int(rot_h * 1.3) + 20

        # 随机位置
        cx = random.randint(CONFIG['margin'] + pad_w // 2, img_w - CONFIG['margin'] - pad_w // 2)
        cy = random.randint(CONFIG['margin'] + pad_h // 2, img_h - CONFIG['margin'] - pad_h // 2)

        color = get_grain_color(cat_info)

        # 绘制基础颗粒
        grain_info = draw_grain_base(None, cx, cy, major, minor, angle, color)
        if grain_info is None:
            continue

        # 检查重叠
        if check_overlap(grain_info['bbox'], existing_bboxes, img_w, img_h):
            continue

        # 根据类别添加特效
        if cat_info['has_crack']:
            grain_info = add_crack_effect(grain_info)
        if cat_info['has_mold']:
            grain_info = add_mold_effect(grain_info)

        # 添加阴影
        grain_info = add_shadow(grain_info)

        # 最终检查
        bb = grain_info['bbox']
        if bb[0] < 0 or bb[1] < 0 or bb[0] + bb[2] > img_w or bb[1] + bb[3] > img_h:
            continue

        grain_info['category_id'] = cat_info['id']
        grain_info['category_name'] = cat_info['name']
        grain_info['angle'] = angle
        grain_info['major'] = major
        grain_info['minor'] = minor
        return grain_info

    return None


def generate_image(img_index, split_name, output_dir):
    """生成一张完整的合成图像"""
    img_w, img_h = CONFIG['img_width'], CONFIG['img_height']
    bg = create_background(img_w, img_h)

    n_grains = random.randint(*CONFIG['grains_per_image'])

    # 确保每张图至少包含2个类别以上
    grain_categories = []
    available_cats = list(CATEGORIES)

    # 强制至少2个类别
    min_cats = min(len(available_cats), random.randint(2, 4))
    mandatory_cats = random.sample(available_cats, min_cats)

    for _ in range(n_grains):
        if len(grain_categories) < min_cats:
            cat = mandatory_cats[len(grain_categories)]
        else:
            cat = random.choice(available_cats)
        grain_categories.append(cat)

    random.shuffle(grain_categories)

    # 放置颗粒
    placed_grains = []
    existing_bboxes = []

    for cat in grain_categories:
        grain = generate_single_grain(cat, img_w, img_h, existing_bboxes)
        if grain:
            placed_grains.append(grain)
            existing_bboxes.append(grain['bbox'])

    # 粘贴所有颗粒到背景
    for grain in placed_grains:
        img = grain['image'].convert('RGBA')
        px, py = grain['x'], grain['y']
        # 只粘贴在画面内的部分
        paste_x = max(0, px)
        paste_y = max(0, py)
        crop_x = paste_x - px
        crop_y = paste_y - py
        crop_w = min(img.width - crop_x, img_w - paste_x)
        crop_h = min(img.height - crop_y, img_h - paste_y)

        if crop_w > 0 and crop_h > 0:
            region = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
            # 使用 alpha 合成
            bg_region = bg.crop((paste_x, paste_y, paste_x + crop_w, paste_y + crop_h)).convert('RGBA')
            blended = Image.alpha_composite(bg_region, region)
            bg.paste(blended.convert('RGB'), (paste_x, paste_y))

    # 保存图片
    filename = f"grain_{split_name}_{img_index:04d}.jpg"
    filepath = os.path.join(output_dir, 'date', 'coco', 'images', split_name, filename)
    bg.save(filepath, 'JPEG', quality=92)

    # 构建 COCO 图像和标注数据
    image_anno = {
        'width': img_w,
        'height': img_h,
        'id': img_index,
        'file_name': f"{split_name}\\{filename}",
    }

    annotations = []
    for i, grain in enumerate(placed_grains):
        bb = grain['bbox']
        annotations.append({
            'id': None,  # 稍后分配
            'image_id': img_index,
            'category_id': grain['category_id'],
            'segmentation': [],
            'bbox': [int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])],
            'ignore': 0,
            'iscrowd': 0,
            'area': int(bb[2] * bb[3]),
        })

    return image_anno, annotations


def main():
    # 解析参数
    parser = argparse.ArgumentParser(description='生成合成谷物颗粒数据集')
    parser.add_argument('--output', default=None, help='输出目录 (默认: 项目目录)')
    parser.add_argument('--train', type=int, default=350, help='训练集数量')
    parser.add_argument('--val', type=int, default=75, help='验证集数量')
    parser.add_argument('--test', type=int, default=75, help='测试集数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--img-width', type=int, default=1920, help='图像宽度')
    parser.add_argument('--img-height', type=int, default=1080, help='图像高度')
    args = parser.parse_args()

    # 更新配置
    CONFIG['n_train'] = args.train
    CONFIG['n_val'] = args.val
    CONFIG['n_test'] = args.test
    CONFIG['seed'] = args.seed
    CONFIG['img_width'] = args.img_width
    CONFIG['img_height'] = args.img_height

    if args.output:
        CONFIG['output_dir'] = args.output
    else:
        CONFIG['output_dir'] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    set_seed(CONFIG['seed'])

    output_dir = CONFIG['output_dir']
    img_w, img_h = CONFIG['img_width'], CONFIG['img_height']

    print('=' * 60)
    print('* 谷物颗粒合成数据集生成器')
    print('=' * 60)
    print(f'  输出目录: {output_dir}')
    print(f'  图像尺寸: {img_w}×{img_h}')
    print(f'  训练集: {CONFIG["n_train"]} 张')
    print(f'  验证集: {CONFIG["n_val"]} 张')
    print(f'  测试集: {CONFIG["n_test"]} 张')
    print(f'  总计: {CONFIG["n_train"] + CONFIG["n_val"] + CONFIG["n_test"]} 张')
    print(f'  随机种子: {CONFIG["seed"]}')
    print()

    # 确保目录存在
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(output_dir, 'date', 'coco', 'images', split)
        os.makedirs(img_dir, exist_ok=True)

    # COCO 基础结构
    coco_base = {
        "info": {
            "year": 2026,
            "version": "1.0",
            "description": "谷物颗粒品质分级检测数据集 - 合成图像 (Grain Quality Grading Detection - Synthetic)",
            "contributor": "Synthetic Generator",
            "url": "",
            "date_created": "2026-06-22 00:00:00.000000",
        },
        "licenses": [
            {"id": 1, "name": "CC BY 4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}
        ],
        "categories": [
            {"id": cat['id'], "name": cat['name'], "supercategory": "grain"}
            for cat in CATEGORIES
        ],
    }

    # 生成数据
    global_ann_id = 0

    for split_name, n_imgs in [('train', CONFIG['n_train']), ('val', CONFIG['n_val']), ('test', CONFIG['n_test'])]:
        print(f'--- 生成 {split_name} 集 ({n_imgs} 张) ---')

        coco_data = coco_base.copy()
        coco_data['images'] = []
        coco_data['annotations'] = []

        for idx in range(n_imgs):
            img_anno, annotations = generate_image(idx + 1, split_name, output_dir)

            # 分配全局标注 ID
            for ann in annotations:
                ann['id'] = global_ann_id
                global_ann_id += 1

            coco_data['images'].append(img_anno)
            coco_data['annotations'].extend(annotations)

            if (idx + 1) % 50 == 0:
                print(f'  已生成: {idx + 1}/{n_imgs}')

        # 保存 COCO JSON
        anno_dir = os.path.join(output_dir, 'date', 'coco', 'annotations')
        os.makedirs(anno_dir, exist_ok=True)
        json_path = os.path.join(anno_dir, f'instances_{split_name}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)

        print(f'  ✅ {split_name}: {len(coco_data["images"])} 图片, {len(coco_data["annotations"])} 标注 → {json_path}')
        print()

    # 总体统计
    print('=' * 60)
    print('📊 数据集概览:')
    total_img = 0
    total_ann = 0
    for split in ['train', 'val', 'test']:
        json_path = os.path.join(output_dir, 'date', 'coco', 'annotations', f'instances_{split}.json')
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        n_img = len(data['images'])
        n_ann = len(data['annotations'])
        total_img += n_img
        total_ann += n_ann
        print(f'  {split}: {n_img} 张图, {n_ann} 个标注')

    print(f'  总计: {total_img} 张图, {total_ann} 个标注')
    print(f'  平均每张: {total_ann / total_img:.1f} 个谷物颗粒')
    print()
    print('✅ 合成数据集生成完成！')
    print()
    print('下一步:')
    print('  1. python scripts/check_dataset.py          # 检查数据集')
    print('  2. python tools/train.py configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py  # 开始训练')


if __name__ == '__main__':
    main()
