"""
一次性脚本：用国内 ModelScope 快源把 bge-small-zh-v1.5 向量模型
下载到项目内永久目录 model_cache/，并写好 fastembed 需要的 .metadata，
之后 fastembed 校验命中就永不重新下载（根治 Temp 被清理后反复慢下载）。
下载完成后本脚本可删除。
"""
import os
import json
import subprocess
import sys

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
REPO_FOLDER = "models--Qdrant--bge-small-zh-v1.5"
REV = "46fbe35fd4374a00fee7de77dfddaeb6dd6a2c59"
REPO = "Qdrant/bge-small-zh-v1.5"

# (文件名, 大小, HF blob_id) —— blob_id 用于 fastembed 的本地校验
FILES = [
    ("config.json", 739, "60938626ad1097a0c1a14be4f8340e32c714a056"),
    ("model_optimized.onnx", 94781076, "bc236a1a7c0fe9066a3cb3c7171170b179746f7d"),
    ("special_tokens_map.json", 125, "a8b3208c2884c4efb86e49300fdd3dc877220cdf"),
    ("tokenizer.json", 439125, "cdb3043fc938fc918c06e66cf704c2ba58f88747"),
    ("tokenizer_config.json", 367, "3a59388f0fd1bd22dec2ce7902c1be8e1fb84107"),
]

snap_dir = os.path.join(CACHE, REPO_FOLDER, "snapshots", REV)
os.makedirs(snap_dir, exist_ok=True)

metadata = {}
for name, size, blob in FILES:
    url = f"https://modelscope.cn/models/{REPO}/resolve/master/{name}"
    dest = os.path.join(snap_dir, name)
    print(f"下载 {name} (期望 {size} 字节) ...", flush=True)
    r = subprocess.run(["curl", "-sL", "--retry", "3", "--max-time", "900", "-o", dest, url])
    actual = os.path.getsize(dest) if os.path.exists(dest) else -1
    if r.returncode != 0 or actual != size:
        print(f"  ✗ 失败 {name}: 返回码={r.returncode} 实际大小={actual} (期望 {size})")
        sys.exit(1)
    metadata[f"snapshots/{REV}/{name}"] = {"size": size, "blob_id": blob}
    print(f"  ✓ 完成 {name}", flush=True)

with open(os.path.join(CACHE, REPO_FOLDER, ".metadata"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"\n已写入 .metadata，共 {len(metadata)} 个文件。")
print(f"缓存目录：{os.path.join(CACHE, REPO_FOLDER)}")
