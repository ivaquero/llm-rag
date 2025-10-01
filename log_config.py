import sys

from loguru import logger

# 移除默认的 handler
logger.remove()

# 添加控制台输出 handler
# 设置格式，并根据日志级别着色
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
# 添加文件输出 handler
# 将日志写入文件，每天生成一个新文件，并自动压缩和清理旧日志
logger.add(
    "rag_project.log",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    encoding="utf-8",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",
)

# 你可以在其他文件中像这样使用 logger:
# from log_config import logger
# logger.info("这是一条信息日志")
# logger.success("这是一条成功日志")
# logger.warning("这是一条警告日志")
# logger.error("这是一条错误日志")
