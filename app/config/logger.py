import logging

# 创建并配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 日志级别
    format='%(asctime)s - %(name)s - [%(levelname)s] - [%(threadName)s-%(thread)d] %(funcName)s - %(message)s',  # 日志格式
    handlers=[
        logging.FileHandler("app.log"),  # 写入日志文件
        logging.StreamHandler()  # 输出到控制台
    ]
)

logger = logging.getLogger("tracker_cat")

def get_logger(name) -> logging.Logger:
    return logging.getLogger(name)