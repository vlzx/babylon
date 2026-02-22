import uvicorn
from fastapi import FastAPI
# 导入我们封装好的超强 logger
from utils.logging import setup_logger

# 1. 初始化日志系统，设置为 DEBUG 级别以便看全所有信息
log = setup_logger(level="DEBUG")

app = FastAPI(title="Rich Loguru Demo")

@app.get("/")
async def root():
    # 演示普通的带颜色标记的输出
    log.info("收到新的请求: 访问了根目录 [bold green]正常[/bold green]")
    log.debug("这是一条 DEBUG 级别的底层信息，用于排查问题...")
    return {"message": "Hello World"}

@app.get("/crash")
async def crash_test():
    # 演示 Rich 极其震撼的异常捕获能力
    log.warning("注意！即将触发一个不可挽回的除零异常！")
    try:
        data = {"user_id": 123, "action": "divide"}
        result = 1 / 0
    except ZeroDivisionError:
        # 使用 log.exception 会自动提取 traceback 并由 Rich 完美渲染
        log.exception(f"程序崩溃了！当时的上下文数据是: {data}")
    return {"status": "failed"}

if __name__ == "__main__":
    log.info("准备启动 FastAPI 服务 [bold cyan]Uvicorn[/bold cyan]...")
    
    # 启动 Uvicorn。
    # 奇迹即将发生：Uvicorn 原生的黑白启动日志和访问日志，
    # 全部都会被我们的拦截器抓住，变成漂亮的 Rich 格式！
    uvicorn.run(app, host="127.0.0.1", port=8000)