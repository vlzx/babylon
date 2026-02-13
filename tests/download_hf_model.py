from huggingface_hub import snapshot_download

repo_id = "Systran/faster-whisper-tiny"
local_dir = "./local_models/faster-whisper-tiny"
print(f'repo_id: {repo_id}')

# 这只会下载文件，不会加载进内存，速度更快
snapshot_download(repo_id=repo_id, local_dir=local_dir)

print("下载完成")