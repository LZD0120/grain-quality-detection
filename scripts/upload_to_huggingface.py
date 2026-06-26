"""
🌾 谷物颗粒品质分级检测数据集 - HuggingFace 上传脚本

用法:
    python scripts/upload_to_huggingface.py --repo YOUR_USERNAME/grain-quality-detection --token hf_xxxxx

获取 token: https://huggingface.co/settings/tokens
"""

import os
import sys
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='上传数据集到 Hugging Face')
    parser.add_argument('--repo', required=True, help='HuggingFace 仓库名 (如: username/grain-quality-detection)')
    parser.add_argument('--token', help='HuggingFace API token (也可设置 HF_TOKEN 环境变量)')
    parser.add_argument('--private', action='store_true', default=False, help='设为私有仓库')
    parser.add_argument('--message', default='Update grain quality grading dataset', help='提交信息')
    return parser.parse_args()


def create_dataset_readme() -> str:
    """生成 HuggingFace Dataset Card"""
    return """---
license: cc-by-4.0
task_categories:
  - object-detection
language:
  - zh
  - en
tags:
  - grain
  - agriculture
  - quality-control
  - object-detection
  - mmdetection
pretty_name: Grain Quality Grading Detection
size_categories:
  - n<1K
---

# 🌾 谷物颗粒品质分级检测数据集 (Grain Quality Grading Detection Dataset)

## 数据集描述

本数据集用于谷物颗粒品质分级的目标检测任务。通过工业相机采集谷物颗粒图像（1920×1080），使用 Label Studio 进行 COCO 格式标注。

## 类别定义

| ID | 类别 | 英文 | 说明 |
|----|------|------|------|
| 0 | 完整粒 | wanzheng | 颗粒完整、无破损、无霉变、色泽正常 |
| 1 | 破损粒 | posun | 颗粒有明显裂纹、断裂、缺损 |
| 2 | 霉变粒 | meibian | 表面有霉斑、菌丝、变色霉变区域 |
| 3 | 异色粒 | yise | 颜色异常（发黄、发黑、发白等），但未霉变 |

## 数据集划分

| 集合 | 图像数 | 标注数 |
|------|--------|--------|
| Train | 350 | 3,416 |
| Val | 75 | -- |
| Test | 75 | -- |

## 使用方法

```python
from datasets import load_dataset

dataset = load_dataset("REPO_ID")
```

## 模型训练

配套训练代码: https://github.com/USERNAME/grain-quality-detection

模型: RTMDet-tiny (MMDetection 3.x)
最佳结果: mAP@0.5 = 1.000, mAP@0.5:0.95 = 0.997

## 引用

```
@dataset{grain_quality_detection,
  author = {Grain Quality Detection Team},
  year = {2026},
  title = {Grain Quality Grading Detection Dataset},
  publisher = {Hugging Face}
}
```
"""


def upload_dataset(repo_id: str, token: str = None, private: bool = False, message: str = "Update dataset"):
    """上传数据集到 Hugging Face Hub"""
    try:
        from huggingface_hub import HfApi, create_repo, upload_folder
    except ImportError:
        print("请先安装: pip install huggingface_hub")
        sys.exit(1)

    # Token 优先级: 参数 > 环境变量
    token = token or os.environ.get('HF_TOKEN')
    if not token:
        print("❌ 请提供 HuggingFace token: --token hf_xxxxx 或设置环境变量 HF_TOKEN")
        print("   获取 token: https://huggingface.co/settings/tokens")
        sys.exit(1)

    api = HfApi(token=token)

    # 创建/确认仓库
    repo_url = f"https://huggingface.co/datasets/{repo_id}"
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            repo_type="dataset",
            exist_ok=True
        )
        print(f"✅ 仓库就绪: {repo_url}")
    except Exception as e:
        print(f"❌ 创建仓库失败: {e}")
        sys.exit(1)

    # 项目根目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 1. 上传标注文件
    annotations_dir = os.path.join(base_dir, 'date', 'coco', 'annotations')
    if os.path.isdir(annotations_dir):
        print(f"📤 上传标注文件...")
        upload_folder(
            folder_path=annotations_dir,
            path_in_repo='annotations',
            repo_id=repo_id,
            token=token,
            repo_type="dataset",
            commit_message=message,
        )
        print(f"   ✅ 标注文件已上传")

    # 2. 上传图像文件 (分批上传，避免超时)
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(base_dir, 'date', 'coco', 'images', split)
        if os.path.isdir(img_dir):
            print(f"📤 上传 {split} 集图像...")
            upload_folder(
                folder_path=img_dir,
                path_in_repo=f'images/{split}',
                repo_id=repo_id,
                token=token,
                repo_type="dataset",
                commit_message=f'{message} - {split} images',
            )
            print(f"   ✅ {split} 图像已上传")

    # 3. 上传 Dataset Card
    readme_content = create_dataset_readme().replace('USERNAME', repo_id.split('/')[0]).replace('REPO_ID', repo_id)
    api.upload_file(
        path_or_fileobj=readme_content.encode('utf-8'),
        path_in_repo='README.md',
        repo_id=repo_id,
        token=token,
        repo_type="dataset",
        commit_message='Update dataset card',
    )

    print()
    print(f"🎉 上传完成: {repo_url}")
    return repo_url


if __name__ == '__main__':
    args = parse_args()
    upload_dataset(args.repo, args.token, args.private, args.message)
