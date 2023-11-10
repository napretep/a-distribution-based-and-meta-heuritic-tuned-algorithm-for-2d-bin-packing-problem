# -*- coding: utf-8 -*-
"""
__project_ = '2d-bin-packing-solver'
__file_name__ = 'update_query.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/11/10 20:44'
"""
import time
import os
import subprocess

POLL_INTERVAL = 8  # 轮询间隔（秒）
LOCAL_REPO_PATH = os.path.split(os.path.abspath(__file__))[0] # 本地仓库路径


def run_git_commands():
    # 设置你的工作目录，即 Git 仓库所在目录

    # 执行 git fetch
    subprocess.check_call(['git', 'fetch', 'origin'], cwd=LOCAL_REPO_PATH)

    # 替换 [branch_name] 为你的目标分支名
    subprocess.check_call(['git', 'reset', '--hard', 'origin/master'], cwd=LOCAL_REPO_PATH)

    print("Git commands executed successfully.")
def main():
    while True:
        run_git_commands()
        print("end pull")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
