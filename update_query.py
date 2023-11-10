# -*- coding: utf-8 -*-
"""
__project_ = '2d-bin-packing-solver'
__file_name__ = 'update_query.py'
__author__ = '十五'
__email__ = '564298339@qq.com'
__time__ = '2023/11/10 20:44'
"""
import requests
import time
import os
import subprocess

# 配置参数
ACCESS_TOKEN ="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" # os.environ.get("GITHUB_TOKEN")  # 从环境变量获取令牌
REPO = "napretep/a-distribution-based-and-meta-heuritic-tuned-algorithm-for-2d-bin-packing-problem"  # GitHub 仓库，格式为'用户名/仓库名'
BRANCH = "main"  # 目标分支
POLL_INTERVAL = 8  # 轮询间隔（秒）
LOCAL_REPO_PATH = os.path.split(os.path.abspath(__file__))[0] # 本地仓库路径

# GitHub API URL
API_URL = f"https://api.github.com/repos/{REPO}/commits/{BRANCH}"

def get_latest_commit_sha():
    headers = {'Authorization': f'token {ACCESS_TOKEN}'}
    response = requests.get(API_URL, headers=headers)
    if response.status_code == 200:
        return response.json()['sha']
    else:
        print("Error fetching the latest commit")
        print("Status Code:", response.status_code)
        print("Response:", response.text)
        return None

def pull_latest_changes():
    subprocess.call(["git", "pull"], cwd=LOCAL_REPO_PATH)

def main():
    last_known_sha = get_latest_commit_sha()
    while True:
        print("Checking for updates...")
        latest_sha = get_latest_commit_sha()

        if latest_sha != last_known_sha:
            print("New commit found. Pulling changes...")
            pull_latest_changes()
            last_known_sha = latest_sha
        else:
            print("No new commits.")

        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
