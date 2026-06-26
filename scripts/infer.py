"""
谷物颗粒品质分级检测 - 推理脚本
Grain Quality Grading Detection - Inference Script

Usage:
    python scripts/infer.py --img test.jpg --config configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py --checkpoint work_dirs/rtmdet_tiny_8xb32-300e_coco/epoch_300.pth
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description='谷物颗粒品质检测 - 单张图片推理')
    parser.add_argument('--img', required=True, help='输入图片路径')
    parser.add_argument('--config', default='configs/rtmdet/my_rtmdet_tiny_8xb32-300e_coco.py', help='配置文件路径')
    parser.add_argument('--checkpoint', required=True, help='模型权重路径 (.pth)')
    parser.add_argument('--out', default='output/result.jpg', help='输出图片路径')
    parser.add_argument('--score-thr', type=float, default=0.3, help='置信度阈值')
    parser.add_argument('--device', default='cuda:0', help='推理设备')
    return parser.parse_args()


def infer_image(img_path, config_path, checkpoint_path, output_path, score_thr=0.3, device='cuda:0'):
    """
    对单张图片进行推理

    类别映射:
        0: wanzheng (完整粒) - 绿
        1: posun    (破损粒) - 红
        2: meibian  (霉变粒) - 灰
        3: yise     (异色粒) - 黄
    """
    from mmdet.apis import init_detector, inference_detector

    # 初始化模型
    model = init_detector(config_path, checkpoint_path, device=device)

    # 推理
    result = inference_detector(model, img_path)

    # 可视化结果
    from mmdet.visualization import DetLocalVisualizer
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = {
        'classes': ('wanzheng', 'posun', 'meibian', 'yise'),
        'palette': [(0, 255, 0), (255, 0, 0), (128, 128, 128), (255, 255, 0)]
    }

    import mmcv
    img = mmcv.imread(img_path)
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        out_file=output_path,
        show=False
    )

    # 打印检测结果
    print(f"\n检测结果 ({os.path.basename(img_path)}):")
    pred_instances = result.pred_instances
    labels = pred_instances.labels.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    bboxes = pred_instances.bboxes.cpu().numpy()
    class_names = ['wanzheng(完整粒)', 'posun(破损粒)', 'meibian(霉变粒)', 'yise(异色粒)']

    for i, (label, score, bbox) in enumerate(zip(labels, scores, bboxes)):
        if score >= score_thr:
            print(f"  [{i+1}] {class_names[int(label)]}: 置信度={score:.3f}, bbox={bbox.astype(int)}")

    print(f"\n结果已保存至: {output_path}")
    return result


if __name__ == '__main__':
    args = parse_args()
    infer_image(args.img, args.config, args.checkpoint, args.out, args.score_thr, args.device)
