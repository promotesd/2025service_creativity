#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from warcio.archiveiterator import ArchiveIterator
from bs4 import BeautifulSoup

WARC_FILE = "/root/CC-MAIN-20180420081400-20180420101400-00000.warc.gz"
OUT_DIR   = "/root/autodl-tmp/dataset/InfiMM-WebMath-40B"

os.makedirs(OUT_DIR, exist_ok=True)

def sanitize_filename(url):
    """将URL转换成文件名的简单函数，去除特殊字符并截断过长"""
    fname = re.sub(r'[^0-9a-zA-Z]+', '_', url)
    return fname[:100]

def main():
    count = 0
    with open(WARC_FILE, "rb") as f:
        for record in ArchiveIterator(f):
            # 只关心 'response' 类型
            if record.rec_type == 'response':
                url = record.rec_headers.get_header('WARC-Target-URI')
                if not url:
                    continue

                # 获取HTTP负载(网页内容)
                raw_bytes = record.content_stream().read()

                # 判断content-type是否HTML, 以免解析二进制资源
                ctype = record.http_headers.get_header('Content-Type') if record.http_headers else ""
                if "html" not in ctype.lower():
                    continue

                # 解析HTML
                soup = BeautifulSoup(raw_bytes, "lxml", from_encoding="utf-8")
                text = soup.get_text(separator="\n")

                # 生成输出文件名
                safe_name = sanitize_filename(url)
                if not safe_name:
                    safe_name = f"page_{count}"

                out_path = os.path.join(OUT_DIR, safe_name + ".txt")

                # 保存到文件
                with open(out_path, "w", encoding="utf-8") as outf:
                    outf.write(f"URL: {url}\n\n")
                    outf.write(text)

                count += 1
                if count % 100 == 0:
                    print(f"[Info] 已处理 {count} 篇网页. 最后一个URL => {url}")

    print(f"完成！共解析 {count} 篇网页，结果保存在: {OUT_DIR}")

if __name__ == "__main__":
    main()
