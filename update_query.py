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

POLL_INTERVAL = 8  # 轮询间隔（秒）
LOCAL_REPO_PATH = os.path.split(os.path.abspath(__file__))[0] # 本地仓库路径



def main():
    while True:
        subprocess.call(["git", "pull"], cwd=LOCAL_REPO_PATH)
        print("end pull")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
