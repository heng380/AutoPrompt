"""
日志收集器：用于收集并实时推送日志到前端
"""
import sys
from typing import List, Dict, Any
from datetime import datetime
import threading


class LogCollector:
    """日志收集器，收集所有 print 输出"""
    
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.listeners: List = []  # SSE 监听器列表
    
    def log(self, message: str, level: str = "info"):
        """添加一条日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "message": message,
            "level": level
        }
        
        with self.lock:
            self.logs.append(log_entry)
            # 通知所有监听器
            for listener in self.listeners:
                try:
                    listener.put(log_entry)
                except Exception as e:
                    # 如果监听器已关闭，从列表中移除
                    self.listeners = [l for l in self.listeners if l != listener]
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """获取所有日志"""
        with self.lock:
            return self.logs.copy()
    
    def clear(self):
        """清空日志"""
        with self.lock:
            self.logs.clear()
    
    def add_listener(self, listener):
        """添加 SSE 监听器"""
        with self.lock:
            self.listeners.append(listener)
    
    def remove_listener(self, listener):
        """移除 SSE 监听器"""
        with self.lock:
            if listener in self.listeners:
                self.listeners.remove(listener)


class QueueListener:
    """队列监听器，用于 SSE"""
    def __init__(self):
        import queue
        self.queue = queue.Queue()
    
    def put(self, log_entry: Dict[str, Any]):
        """添加日志到队列"""
        try:
            self.queue.put(log_entry, block=False)
        except:
            pass
    
    def get(self, timeout=None):
        """从队列获取日志"""
        try:
            return self.queue.get(timeout=timeout)
        except:
            return None


# 全局日志收集器实例
_log_collector = LogCollector()


def get_log_collector() -> LogCollector:
    """获取全局日志收集器"""
    return _log_collector


def log_print(*args, **kwargs):
    """替换 print 函数，将输出收集到日志收集器"""
    message = ' '.join(str(arg) for arg in args)
    _log_collector.log(message, level=kwargs.get('level', 'info'))
    # 同时输出到原始 stdout（可选）
    print(*args, **kwargs, file=sys.stdout)

