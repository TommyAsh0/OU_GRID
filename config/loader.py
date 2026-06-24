# ====================================================================
# 配置加载器
#
# 作用：
#   读取 config/settings.yaml，返回一个嵌套字典供全项目使用。
#   提供一个全局单例 CONFIG，避免每个模块都重复读文件。
#
# 用法：
#   from config.loader import load_config, CONFIG
#   cfg = load_config()              # 显式加载
#   ma_len = CONFIG["indicators"]["ma_length"]   # 直接用单例
# ====================================================================

import os
import yaml

# 本文件所在目录，即项目的 config/ 目录。
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
# settings.yaml 的绝对路径。
_SETTINGS_PATH = os.path.join(_CONFIG_DIR, "settings.yaml")
# 项目根目录（config/ 的上一级），用于把配置中的相对路径转成绝对路径。
PROJECT_ROOT = os.path.dirname(_CONFIG_DIR)


def load_config(path: str = _SETTINGS_PATH) -> dict:
    """读取 YAML 配置文件并返回字典。

    Args:
        path: 配置文件路径，默认为 config/settings.yaml。

    Returns:
        dict: 解析后的配置字典（与 YAML 结构一一对应）。
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> str:
    """把相对项目根目录的路径转换为绝对路径。

    配置文件里写的是 "data/raw" 这类相对路径，
    用本函数转成绝对路径后再使用，保证在任何工作目录下都能正确定位。

    Args:
        relative_path: 相对项目根目录的路径。

    Returns:
        str: 绝对路径。
    """
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(PROJECT_ROOT, relative_path)


# 全局配置单例：导入本模块时自动加载一次。
CONFIG = load_config()
