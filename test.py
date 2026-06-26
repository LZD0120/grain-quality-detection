"""
🌾 谷物颗粒品质分级检测 - 推理/评估脚本
   在谷物项目目录下直接运行:
      python test.py --checkpoint work_dirs/rtmdet_tiny_8xb32-300e_coco/epoch_300.pth

   单张图片推理:
      python test.py --checkpoint work_dirs/rtmdet_tiny_8xb32-300e_coco/epoch_20.pth --img xxx.jpg
"""

import sys
import os
import subprocess
import argparse
import glob

# ── 所有文件都在本项目内，无需外部路径 ──────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(PROJECT_DIR, 'configs', 'rtmdet', 'my_rtmdet_tiny_8xb32-300e_coco.py')
TEST_SCRIPT = os.path.join(PROJECT_DIR, 'tools', 'test.py')


def main():
    parser = argparse.ArgumentParser(description='谷物颗粒品质分级检测 - 推理/评估')
    parser.add_argument('--config', default=DEFAULT_CONFIG, help='配置文件路径')
    parser.add_argument('--checkpoint', help='模型权重路径 (.pth)')
    parser.add_argument('--img', help='单张图片推理（可选）')
    parser.add_argument('--out', default='output/result.jpg', help='推理结果输出路径')
    parser.add_argument('--score-thr', type=float, default=0.3, help='置信度阈值')
    args = parser.parse_args()

    os.chdir(PROJECT_DIR)

    # 自动查找最新 checkpoint
    if not args.checkpoint:
        ckpt_dir = os.path.join(PROJECT_DIR, 'work_dirs', 'rtmdet_tiny_8xb32-300e_coco')
        if os.path.isdir(ckpt_dir):
            ckpts = glob.glob(os.path.join(ckpt_dir, 'epoch_*.pth'))
            if ckpts:
                args.checkpoint = sorted(ckpts)[-1]
                print(f'自动选择 checkpoint: {args.checkpoint}')

    if not args.checkpoint:
        print('❌ 未找到 checkpoint，请通过 --checkpoint 指定')
        return

    if args.img:
        # 单张推理
        from mmdet.apis import init_detector, inference_detector
        import mmcv

        model = init_detector(args.config, args.checkpoint, device='cuda:0')
        result = inference_detector(model, args.img)
        img = mmcv.imread(args.img)

        from mmdet.visualization import DetLocalVisualizer
        visualizer = DetLocalVisualizer()
        visualizer.dataset_meta = {
            'classes': ('wanzheng', 'posun', 'meibian', 'yise'),
            'palette': [(0, 255, 0), (255, 0, 0), (128, 128, 128), (255, 255, 0)]
        }
        visualizer.add_datasample('result', img, data_sample=result,
                                  draw_gt=False, out_file=args.out)
        print(f'✅ 推理完成 → {args.out}')

        # 打印结果
        pred = result.pred_instances
        names = ['wanzheng', 'posun', 'meibian', 'yise']
        for i, (label, score, bbox) in enumerate(zip(
                pred.labels.cpu().numpy(),
                pred.scores.cpu().numpy(),
                pred.bboxes.cpu().numpy())):
            if score >= args.score_thr:
                print(f'  [{i+1}] {names[int(label)]}: score={score:.3f}, bbox={bbox.astype(int)}')
    else:
        # 测试集评估
        cmd = [sys.executable, TEST_SCRIPT, args.config, args.checkpoint]
        print(f'▶ 执行: {" ".join(cmd)}')
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        subprocess.run(cmd, env=env)


if __name__ == '__main__':
    main()
