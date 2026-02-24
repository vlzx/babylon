import soundfile as sf
import time
import numpy as np

def stream_wav_realtime(wav_path, frame_duration_ms=20, dtype="float32"):
    """
    使用 soundfile 按实际时间速率模拟读取并输出音频。
    """
    # 获取音频的基础信息
    info = sf.info(wav_path)
    sample_rate = info.samplerate
    
    # 计算每次需要读取的帧数
    frame_duration_sec = frame_duration_ms / 1000.0
    chunk_size = int(sample_rate * frame_duration_sec)
    
    print(f"音频信息: 采样率 {sample_rate}Hz, 声道 {info.channels}, 时长 {info.duration:.2f}秒")
    print(f"每 {frame_duration_ms}ms 读取 {chunk_size} 帧")

    # 记录开始处理的绝对时间
    start_time = time.time()
    chunks_processed = 0
    
    # sf.blocks 是一个生成器，专门用于分块读取大文件
    # dtype='int16': 输出 16-bit 整数类型的 numpy 数组 (常用的音频格式)
    # dtype='float32': 如果你是要送入 PyTorch/TensorFlow 模型，通常改为 float32
    # fill_value=0: 确保最后一块如果不足 20ms，自动用静音(0)补齐长度
    block_generator = sf.blocks(
        wav_path, 
        blocksize=chunk_size, 
        dtype=dtype, 
        fill_value=0
    )
    
    for audio_chunk in block_generator:
        # 1. 计算当前这个 Chunk 应该在什么时间点被输出
        target_time = start_time + (chunks_processed * frame_duration_sec)
        
        # 2. 获取当前时间，如果还没到目标时间，就精准睡眠补齐差值
        current_time = time.time()
        if current_time < target_time:
            time.sleep(target_time - current_time)
        
        if chunks_processed == 0:
            print('首包音频输出')
        
        yield audio_chunk
        
        chunks_processed += 1
        
    print("音频流处理完毕。")

# 使用示例
if __name__ == "__main__":
    # 请替换为你的音频文件路径
    # stream_wav_realtime("test_audio.wav", frame_duration_ms=20)
    pass