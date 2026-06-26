# ============================================================
# GitHub 上传脚本 (PowerShell - Windows)
# 谷物颗粒品质分级检测数据集
# ============================================================

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " 谷物颗粒品质分级检测 - GitHub 上传脚本" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# ============================================================
# 配置区域 - 请修改以下变量
# ============================================================
$GITHUB_USERNAME = "LZD0120"
$REPO_NAME = "grain-quality-detection"
$REPO_DESCRIPTION = "谷物颗粒品质分级检测数据集 - 基于MMDetection+RTMDet"
# ============================================================

# 进入项目根目录
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

# 初始化 Git 仓库 (如果还没有)
if (-not (Test-Path ".git")) {
    Write-Host ""
    Write-Host "[1/4] 初始化 Git 仓库..." -ForegroundColor Yellow
    git init
    git checkout -b main
} else {
    Write-Host ""
    Write-Host "[1/4] Git 仓库已存在，跳过初始化" -ForegroundColor Yellow
}

# 添加文件
Write-Host ""
Write-Host "[2/4] 添加文件到暂存区..." -ForegroundColor Yellow

git add README.md
git add 实验报告.md
git add .gitignore
git add configs/
git add scripts/

# 数据集标注文件 (图片由 .gitignore 排除，需单独上传到 HuggingFace)
git add date/coco/annotations/

# 提交
Write-Host ""
Write-Host "[3/4] 创建提交..." -ForegroundColor Yellow
git commit -m "feat: 谷物颗粒品质分级检测数据集 - 初始版本

- 4类别COCO格式标注 (完整粒/破损粒/霉变粒/异色粒)
- RTMDet-tiny 训练配置
- 实验报告模板
- 数据校验/转换/上传工具脚本

数据集上传至 HuggingFace: https://huggingface.co/datasets/$GITHUB_USERNAME/$REPO_NAME"

# 关联远程仓库并推送
Write-Host ""
Write-Host "[4/4] 推送到 GitHub..." -ForegroundColor Yellow
git remote remove origin 2>$null
git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
git push -u origin main

# 创建标签
Write-Host ""
Write-Host "创建版本标签 v1.0..." -ForegroundColor Yellow
git tag -a v1.0 -m "v1.0: 初始数据集发布，包含4类谷物品质标注"
git push origin v1.0

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host " ✅ 上传完成!" -ForegroundColor Green
Write-Host "   仓库地址: https://github.com/$GITHUB_USERNAME/$REPO_NAME" -ForegroundColor Green
Write-Host "   数据集地址: https://huggingface.co/datasets/$GITHUB_USERNAME/$REPO_NAME" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
