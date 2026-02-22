import os
import sys
import logging
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler

def setup_logger(level="INFO", log_dir="logs", show_path=True):
    """
    配置并返回一个融合了 Rich 视觉效果的 Loguru 对象，并支持文件持久化
    
    :param level: 最低日志级别
    :param log_dir: 日志文件存储目录。如果为 None，则不写入文件。
    :param show_path: 控制台是否显示代码路径
    """
    # 1. 清除 Loguru 默认处理器
    logger.remove()

    # ==========================================
    # 2. 配置终端输出 (Rich 华丽渲染)
    # ==========================================
    _console = Console(force_terminal=True, width=120)
    logger.add(
        RichHandler(
            console=_console, 
            rich_tracebacks=True,
            markup=True,
            show_path=show_path
        ),
        level=level,
        format="{message}", 
    )

    # ==========================================
    # 3. 配置文件输出 (纯文本持久化)
    # ==========================================
    if log_dir:
        # 自动创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        # 日志文件路径模式
        log_path = os.path.join(log_dir, "{time:YYYY-MM-DD}.log")
        
        logger.add(
            log_path,
            level=level,
            # 文件日志的格式：包含完整时间、级别、模块名、行号和消息
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",       # 切割：文件达到 10MB 时自动创建新文件
            retention="30 days",    # 清理：保留最近 30 天的日志，自动清理旧文件
            compression="zip",      # 压缩：历史日志自动压缩为 zip 节省空间
            encoding="utf-8",       # 确保中文不乱码
            enqueue=True,           # 异步写入：保证多线程/多进程下的安全与高性能
        )

    # ==========================================
    # 4. 深度拦截原生 logging (Uvicorn/第三方库)
    # ==========================================
    class InterceptHandler(logging.Handler):
        def emit(self, record):
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

            frame, depth = logging.currentframe(), 2
            while frame and frame.f_code.co_filename == logging.__file__:
                frame = frame.f_back
                depth += 1

            logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # 清空第三方库自带的处理器，强制向上传递
    for name in logging.root.manager.loggerDict.keys():
        temp_logger = logging.getLogger(name)
        temp_logger.handlers.clear()
        temp_logger.propagate = True
    
    return logger

# 预设一个默认的 logger 实例
# 运行后会在项目根目录自动生成一个 'logs' 文件夹
logger = setup_logger(level="INFO", log_dir="logs")