"""
🌾 谷物颗粒品质分级检测 - GitHub 上传脚本

用法:
    python scripts/upload_to_github.py --repo YOUR_USERNAME/grain-quality-detection

前置条件:
    1. 安装 Git: https://git-scm.com/download/win
    2. Git 首次配置:
       git config --global user.name "你的名字"
       git config --global user.email "你的邮箱"

认证方式 (任选其一):
    A) GitHub CLI (推荐):
       winget install GitHub.cli
       gh auth login
    B) Personal Access Token:
       https://github.com/settings/tokens → 生成 token
       git remote set-url origin https://TOKEN@github.com/USERNAME/grain-quality-detection.git
    C) SSH Key:
       https://docs.github.com/en/authentication/connecting-to-github-with-ssh
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='上传项目代码到 GitHub')
    parser.add_argument('--repo', required=True, help='GitHub 仓库名 (如: username/grain-quality-detection)')
    parser.add_argument('--token', help='GitHub Personal Access Token')
    parser.add_argument('--private', action='store_true', default=False, help='设为私有仓库')
    parser.add_argument('--message', default='feat: 谷物颗粒品质分级检测 - 初始版本', help='提交信息')
    return parser.parse_args()


def run(cmd, cwd=None, env=None):
    """执行命令并返回 (returncode, stdout, stderr)"""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd, env=env,
        capture_output=True, text=True, encoding='utf-8', errors='replace'
    )
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def check_git():
    """检查 Git 是否可用"""
    code, stdout, stderr = run('git --version')
    if code != 0:
        print("❌ Git 未安装或不在 PATH 中")
        print("   请安装: https://git-scm.com/download/win")
        return False
    print(f"✅ {stdout}")
    return True


def create_github_repo(repo_id: str, token: str = None, private: bool = False):
    """通过 GitHub API 创建仓库"""
    owner, repo_name = repo_id.split('/')

    if token:
        import urllib.request
        import json

        data = json.dumps({
            'name': repo_name,
            'private': private,
            'description': '🌾 谷物颗粒品质分级检测 - 基于 MMDetection + RTMDet-tiny (mAP@0.5=1.000)',
        }).encode('utf-8')

        req = urllib.request.Request(
            'https://api.github.com/user/repos',
            data=data,
            headers={
                'Authorization': f'token {token}',
                'Accept': 'application/vnd.github+json',
                'Content-Type': 'application/json',
            },
            method='POST'
        )

        try:
            with urllib.request.urlopen(req) as resp:
                result = json.loads(resp.read())
                print(f"✅ GitHub 仓库已创建: {result['html_url']}")
                return result['html_url']
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            if 'already exists' in body.lower() or e.code == 422:
                print(f"⚠ 仓库已存在: https://github.com/{repo_id}")
                return f"https://github.com/{repo_id}"
            print(f"❌ 创建仓库失败: {e.code} - {body}")
            return None
    else:
        print("⚠ 未提供 token，跳过 GitHub API 创建仓库")
        print("   请手动创建: https://github.com/new")
        return None


def git_init_and_push(project_dir: str, repo_id: str, token: str = None, message: str = "Initial commit"):
    """Git 初始化、提交并推送"""
    owner, repo_name = repo_id.split('/')

    # 构建远程 URL
    if token:
        remote_url = f"https://{token}@github.com/{repo_id}.git"
    else:
        remote_url = f"https://github.com/{repo_id}.git"

    os.chdir(project_dir)

    # 1. Git Init
    if not os.path.isdir('.git'):
        code, _, stderr = run('git init')
        if code != 0:
            print(f"❌ git init 失败: {stderr}")
            return False
        run('git checkout -b main')
        print("✅ Git 仓库已初始化")
    else:
        print("✅ Git 仓库已存在")

    # 2. 添加文件
    print("📦 添加文件到暂存区...")
    git_add_list = [
        'README.md',
        '实验报告.md',
        '.gitignore',
        'train.py',
        'test.py',
        'configs/',
        'scripts/',
        'tools/',
        'mmdet/',
    ]
    for path in git_add_list:
        full = os.path.join(project_dir, path)
        if os.path.exists(full):
            code, _, stderr = run(f'git add "{path}"')
            if code != 0 and 'did not match any files' not in stderr:
                print(f"   ⚠ git add {path}: {stderr}")

    print("   ✅ 文件已添加")

    # 3. 提交
    print("📝 创建提交...")
    code, _, stderr = run(f'git commit -m "{message}"')
    if code != 0:
        if 'nothing to commit' in stderr:
            print("   ⚠ 没有新变更需要提交")
        else:
            print(f"   ⚠ git commit: {stderr}")

    # 4. 设置远程并推送
    print("🚀 推送到 GitHub...")
    run(f'git remote remove origin')
    run(f'git remote add origin "{remote_url}"')
    code, stdout, stderr = run('git push -u origin main --force')
    if code != 0:
        print(f"❌ 推送失败: {stderr}")
        print()
        print("💡 可能的原因和解决方案:")
        print("   1. 仓库不存在 → 先创建: https://github.com/new")
        print("   2. 认证失败 → 使用: scripts/upload_to_github.py --token ghp_xxxxx")
        print("   3. 网络问题 → 检查代理或VPN")
        return False

    print(f"   ✅ 推送成功")

    # 5. 创建版本标签
    run('git tag -a v1.0 -m "v1.0: 初始数据集发布，4类谷物品质标注，mAP@0.5=1.000"')
    run('git push origin v1.0')

    return True


def main():
    args = parse_args()
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print('=' * 60)
    print('  🌾 谷物颗粒品质分级检测 - GitHub 上传')
    print('=' * 60)
    print(f'  项目目录: {project_dir}')
    print(f'  目标仓库: https://github.com/{args.repo}')
    print('=' * 60)
    print()

    # 1. 检查 Git
    if not check_git():
        sys.exit(1)

    # 2. 创建 GitHub 仓库
    repo_url = create_github_repo(args.repo, args.token, args.private)
    if not repo_url:
        print("请手动创建仓库后重试，或提供 --token 参数")
        sys.exit(1)

    # 3. Git 初始化、提交、推送
    success = git_init_and_push(project_dir, args.repo, args.token, args.message)

    # 4. 结果
    print()
    print('=' * 60)
    if success:
        print(f'  ✅ 上传完成!')
        print(f'  📂 GitHub: https://github.com/{args.repo}')
    else:
        print(f'  ❌ 上传失败，请检查上述错误信息')
    print('=' * 60)


if __name__ == '__main__':
    main()
