"""
工具模块
"""
from .logger import get_log_collector, log_print, LogCollector

__all__ = ['get_log_collector', 'log_print', 'LogCollector']

# 从父目录的 utils.py 导入（为了兼容性）
# 注意：这些函数实际上在 utils.py 中，不在 utils 包中
# 在 app.py 中应该直接使用 from utils import ... 或者 import utils

