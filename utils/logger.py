# utils/logger.py
"""日志工具"""
import os
import sys
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """统一日志管理"""

    def __init__(self, log_dir: str, name: str = 'DinoDefectClassifier'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # 控制台输出
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_fmt = logging.Formatter(
            '[%(asctime)s][%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        console_handler.setFormatter(console_fmt)
        self.logger.addHandler(console_handler)

        # 文件输出
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f'train_{timestamp}.log')
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_fmt)
        self.logger.addHandler(file_handler)

        # TensorBoard
        self.writer = SummaryWriter(
            os.path.join(log_dir, 'tensorboard', timestamp)
        )

    def info(self, msg: str):
        self.logger.info(msg)

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def add_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def add_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def add_image(self, tag: str, img_tensor, step: int):
        self.writer.add_image(tag, img_tensor, step)

    def close(self):
        self.writer.close()