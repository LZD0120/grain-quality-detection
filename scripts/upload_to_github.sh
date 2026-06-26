#!/bin/bash
# ============================================================
# GitHub 上传脚本
# 谷物颗粒品质分级检测数据集
# ============================================================

set -e

echo "=========================================="
echo " 谷物颗粒品质分级检测 - GitHub 上传脚本"
echo "=========================================="

# ============================================================
# 配置区域 - 请修改以下变量
# ============================================================
GITHUB_USERNAME="YOUR_USERNAME"
REPO_NAME="grain-quality-detection"
REPO_DESCRIPTION="谷物颗粒品质分级检测数据集 - 基于MMDetection+RTMDet"
# ============================================================

# 初始化 Git 仓库 (如果还没有)
if [ ! -d ".git" ]; then
    echo ""
    echo "[1/4] 初始化 Git 仓库..."
    git init
    git checkout -b main
else
    echo ""
    echo "[1/4] Git 仓库已存在，跳过初始化"
fi

# 添加文件
echo ""
echo "[2/4] 添加文件到暂存区..."

# 代码和配置文件
git add README.md
git add 实验报告.md
git add .gitignore
git add configs/
git add scripts/

# 数据集标注文件 (图片由 .gitignore 排除，需单独上传到 HuggingFace)
git add date/coco/annotations/

# 提交
echo ""
echo "[3/4] 创建提交..."
git commit -m "feat: 谷物颗粒品质分级检测数据集 - 初始版本

- 4类别COCO格式标注 (完整粒/破损粒/霉变粒/异色粒)
- RTMDet-tiny 训练配置
- 实验报告模板
- 数据校验/转换/上传工具脚本

数据集上传至 HuggingFace: https://huggingface.co/datasets/$GITHUB_USERNAME/$REPO_NAME"

# 关联远程仓库并推送
echo ""
echo "[4/4] 推送到 GitHub..."
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
git push -u origin main

# 创建标签
echo ""
echo "创建版本标签 v1.0..."
git tag -a v1.0 -m "v1.0: 初始数据集发布，包含4类谷物品质标注"
git push origin v1.0

echo ""
echo "=========================================="
echo " ✅ 上传完成!"
echo "   仓库地址: https://github.com/$GITHUB_USERNAME/$REPO_NAME"
echo "   数据集地址: https://huggingface.co/datasets/$GITHUB_USERNAME/$REPO_NAME"
echo "=========================================="
