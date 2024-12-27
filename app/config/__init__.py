import builtins

from app.config.logger import get_logger

def init_config():
    builtins.get_logger = get_logger
    log = get_logger("config")
    log.info('Initializing config')