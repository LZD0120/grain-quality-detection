"""
🌾 谷物颗粒品质分级检测 - 独立训练脚本
   在谷物项目目录下直接运行:
      python train.py

   从 checkpoint 恢复训练:
      python train.py --resume
"""

import sys
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# ★ 确保项目目录在最前面，这样才能找到项目内的 mmdet/
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

os.chdir(PROJECT_DIR)

# Windows UTF-8 编码——必须最先设置
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'


def main():
    import argparse

    parser = argparse.ArgumentParser(description='谷物颗粒品质分级检测 - 训练')
    parser.add_argument('--config', default='configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py',
                        help='配置文件路径')
    parser.add_argument('--resume', action='store_true', help='恢复训练')
    args = parser.parse_args()

    # 直接在进程内运行，避免子进程的编码/路径问题
    sys.argv = [
        sys.argv[0],           # 脚本名
        args.config,
    ]
    if args.resume:
        sys.argv.append('--resume')

    print('=' * 65)
    print('  🌾 谷物颗粒品质分级检测 - RTMDet-tiny 训练')
    print('=' * 65)
    print(f'  项目目录 : {PROJECT_DIR}')
    print(f'  配置文件 : {args.config}')
    print(f'  类别     : wanzheng, posun, meibian, yise')
    print(f'  工作目录 : work_dirs/rtmdet_tiny_8xb32-300e_coco/')
    print('=' * 65)

    # 直接调用 mmdet tools/train.py 的 main
    from tools.train import main as train_main
    train_main()


if __name__ == '__main__':
    main()
