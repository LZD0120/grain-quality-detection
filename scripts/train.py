"""
谷物颗粒品质分级检测 - 训练脚本
Grain Quality Grading Detection - Training Script

使用 RTMDet-tiny 模型在 COCO 格式数据集上进行训练
Usage:
    python scripts/train.py
    或
    python tools/train.py configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py
"""

import os
import sys

# 添加 mmdetection 路径(如果使用独立的 mmdetection 仓库)
# sys.path.insert(0, 'mmdetection-main')

# ============================================================
# 方式一: 使用命令行训练 (推荐)
# ============================================================
# python tools/train.py configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py

# ============================================================
# 方式二: 使用 Python API 训练
# ============================================================
def train_with_api():
    """
    使用 mmdet 的 Python API 进行训练
    需要安装 mmdet: pip install mmdet
    """
    from mmengine.config import Config
    from mmdet.utils import register_all_modules

    # 注册所有 mmdet 模块
    register_all_modules()

    # 加载配置
    cfg = Config.fromfile('configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py')

    # 根据实际 GPU 数量调整
    # 单 GPU 训练
    from mmengine.runner import Runner
    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    print("=" * 60)
    print("谷物颗粒品质分级检测 - RTMDet 训练")
    print("=" * 60)
    print(f"当前工作目录: {os.getcwd()}")
    print()
    print("请使用以下命令开始训练:")
    print("  python tools/train.py configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py")
    print()
    print("或使用多 GPU 训练:")
    print("  bash tools/dist_train.sh configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py 2")
