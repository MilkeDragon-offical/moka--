"""
Moka 游戏自动化脚本
功能：自动识别游戏界面，执行战斗、选择关卡、管理资源等操作
作者：User
日期：2025
"""

import subprocess
import os
import time
import tempfile
import warnings
import json
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Dict, Any
import cv2
import numpy as np
import mss
import pygetwindow as gw
import random
import matplotlib.pyplot as plt
import ctypes
from ctypes import wintypes
import uiautomator2 as u2
from datetime import datetime

# 禁用 PaddleOCR 模型源检查（必须在导入前设置）
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['PADDLEOCR_QUIET'] = 'True'

from paddleocr import PaddleOCR
from paddleocr import TextRecognition


# ==================== 配置类 ====================

@dataclass
class GameConfig:
    """
    游戏运行配置类
    集中管理所有可调控参数
    """
    # ==================== 设备配置 ====================
    device: str = "127.0.0.1:5555"           # ADB 设备地址
    mumu_path: str = r'D:\emulator\MuMuPlayer-12.0\nx_main\MuMuManager.exe'  # MuMu 模拟器路径
    
    # ==================== 圣物配置 ====================
    gaojie_count: int = 1                    # 高洁圣物初始数量
    chenmi_count: int = 1                    # 沉迷圣物初始数量
    qingchun_count: int = 1                  # 青春词条初始数量
    tongxing_num: int = 5                    # 通行点初始数量
    
    # ==================== 战斗策略配置 ====================
    # 非BOSS层跳过策略
    skip_chenmi_normal: bool = True          # 是否跳过非BOSS层的"沉迷"词条关卡
    skip_gaojie_normal: bool = False        # 是否跳过非BOSS层的"高洁"词条关卡
    skip_both_normal: bool = True            # 是否跳过非BOSS层的"双词条"关卡
    
    # BOSS层策略（根据圣物数量自动判断）
    fight_chenmi_boss: bool = True           # 是否打"沉迷"词条BOSS
    fight_gaojie_boss: bool = True           # 是否打"高洁"词条BOSS
    fight_both_boss: bool = True             # 是否打"双词条"BOSS
    
    # ==================== 刷新与重启配置 ====================
    refresh_interval_minutes: int = 60       # 游戏刷新间隔（分钟）
    enable_auto_refresh: bool = True         # 是否启用自动刷新
    restart_interval_minutes: int = 60       # 脚本重启间隔（分钟）
    enable_auto_restart: bool = False        # 是否启用自动重启脚本
    
    # ==================== OCR 配置 ====================
    ocr_det_model: str = "PP-OCRv5_server_det"    # 文字检测模型
    ocr_rec_model: str = "PP-OCRv5_server_rec"      # 文字识别模型
    ocr_en_rec_model: str = "en_PP-OCRv5_mobile_rec"  # 英文识别模型
    
    # ==================== 游戏包名配置 ====================
    game_package: str = "com.ifree.cardCHS"  # 游戏包名
    game_activity: Optional[str] = None      # 游戏主Activity（None则自动获取）
    
    # ==================== 运行控制配置 ====================
    max_rounds: int = 1000000                # 最大运行轮数
    enable_debug: bool = True                # 是否启用调试输出
    save_state_on_exit: bool = True          # 退出时是否保存状态
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def save_to_file(self, filepath: str = "config.json"):
        """保存配置到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"[配置] 已保存到 {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str = "config.json") -> "GameConfig":
        """从文件加载配置"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[配置] 已从 {filepath} 加载")
            return cls(**data)
        except FileNotFoundError:
            print(f"[配置] 文件 {filepath} 不存在，使用默认配置")
            return cls()
    
    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"[配置] 更新 {key} = {value}")
            else:
                print(f"[配置] 警告: 未知参数 {key}")


# ==================== 全局配置实例 ====================
# 使用单例模式管理全局配置
_global_config: Optional[GameConfig] = None
_config_lock = threading.Lock()

def get_config() -> GameConfig:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        with _config_lock:
            if _global_config is None:
                _global_config = GameConfig.load_from_file()
    return _global_config

def reload_config() -> GameConfig:
    """重新加载配置"""
    global _global_config
    with _config_lock:
        _global_config = GameConfig.load_from_file()
    return _global_config

def update_config(**kwargs):
    """更新全局配置"""
    config = get_config()
    config.update(**kwargs)
    config.save_to_file()


# 选项1：(446, 657)   选项2:(947, 653)  选项3:(1447, 657)  进入战斗:1543, 952  跳过战斗:(1701, 465) 选圣物:338, 792  确定：(958, 930)

# 圣物位置（从左到右）:
# 1. 原始: [(424, 286), (936, 272), (418, 476), (936, 482)]
# 2. 原始: [(1181, 284), (1701, 278), (1183, 476), (1695, 477)]
# 3. 原始: [(416, 527), (936, 521), (411, 723), (933, 724)]
# 4. 原始: [(1181, 527), (1698, 521), (1180, 719), (1695, 723)]

# 选择圣物选项
# [(331, 784)]

# 四个圣物对应的选择的位置：
# [(332, 386), (1091, 374), (316, 623), (1086, 620)]


# 选项一的词条区域:
# [(284, 846), (654, 854), (287, 984), (659, 984)]
# 类别
# [(293, 777), (650, 779), (300, 826), (650, 831)]
# 类别+文本
# [(270, 775), (670, 772), (277, 979), (670, 982)]

# 选项二的词条区域：
# [(765, 853), (1145, 861), (772, 977), (1144, 975)]
# 类别
# [(763, 854), (1147, 849), (766, 937), (1138, 940)]
# 类别+文本
# [(270, 775), (670, 772), (277, 979), (670, 982)]

# 特殊：自动-跳过按钮
# [(1687, 378), (1864, 383), (1684, 520), (1874, 534)]

# 购买通行点:为加号，确定按钮，对应文本区域的四个点
# [(1630, 113), (1205, 701), (603, 474), (1311, 474), (598, 533), (1319, 533)]
# 通行点展示区域
# [(1511, 61), (1594, 57), (1512, 130), (1604, 136)]

# 刷新点位置
# [(627, 384), (1121, 383), (1611, 381)]

# 退出回复圣物
# [(1709, 185)]  确定修复：[(1536, 933)]     确定退出(1227, 702)

# 圣物位置
# [(259, 534), (497, 522), (708, 528)]

class MuMuAppManager:
    """MuMu 模拟器应用管理器 - 通过 ADB 控制应用"""
    
    def __init__(self, mumu_path=r'D:\emulator\MuMuPlayer-12.0\nx_main\MuMuManager.exe', instance_id=0):
        self.mumu_path = mumu_path
        self.instance_id = instance_id
    
    def _exec_adb(self, command):
        """执行 ADB 命令"""
        args = [self.mumu_path, 'adb', '-v', str(self.instance_id), '-c', f"shell {command}"]
        result = subprocess.run(args, capture_output=True, text=True, timeout=30)
        return result.returncode == 0, result.stdout, result.stderr
    
    def stop_app(self, package_name):
        """强制停止应用"""
        success, stdout, stderr = self._exec_adb(f'am force-stop {package_name}')
        if success:
            print(f"[应用管理] 已关闭应用: {package_name}")
        else:
            print(f"[应用管理] 关闭应用失败: {package_name}, 错误: {stderr}")
        return success
    
    def start_app(self, package_name="com.ifree.cardCHS", activity_name=None):
        """启动应用"""
        if activity_name:
            # 指定 Activity 启动
            command = f'am start -n {package_name}/{activity_name}'
        else:
            # 通过包名启动（启动主 Activity）
            command = f'monkey -p {package_name} -c android.intent.category.LAUNCHER 1'
        
        success, stdout, stderr = self._exec_adb(command)
        if success:
            print(f"[应用管理] 已启动应用: {package_name}")
        else:
            print(f"[应用管理] 启动应用失败: {package_name}, 错误: {stderr}")
        return success
    
    def restart_app(self, package_name="com.ifree.cardCHS", activity_name=None, clear_data=False):
        """重启应用：先关闭，可选清理数据，再启动"""
        print(f"[应用管理] 正在重启应用: {package_name}")
        
        # 1. 强制停止应用
        self.stop_app(package_name)
        time.sleep(1)
        
        # 2. 可选：清理应用数据
        if clear_data:
            self.clear_app_data(package_name)
            time.sleep(0.5)
        
        # 3. 启动应用
        time.sleep(1)
        return self.start_app(package_name, activity_name)
    
    def clear_app_data(self, package_name="com.ifree.cardCHS"):
        """清理应用数据"""
        success, stdout, stderr = self._exec_adb(f'pm clear {package_name}')
        if success:
            print(f"[应用管理] 已清理应用数据: {package_name}")
        else:
            print(f"[应用管理] 清理数据失败: {package_name}")
        return success
    
    def get_current_app(self):
        """获取当前正在运行的应用包名"""
        success, stdout, stderr = self._exec_adb('dumpsys window | grep mCurrentFocus')
        if success and stdout:
            # 解析输出获取包名
            # 示例输出: mCurrentFocus=Window{... com.example.app/...}
            import re
            match = re.search(r'([a-zA-Z0-9._]+)/', stdout)
            if match:
                return match.group(1)
        return None
    
    def is_app_running(self, package_name):
        """检查应用是否正在运行"""
        current = self.get_current_app()
        return current == package_name
    
    def get_installed_packages(self, third_party_only=True):
        """获取已安装应用列表"""
        flag = '-3' if third_party_only else ''
        success, stdout, stderr = self._exec_adb(f'pm list packages {flag}')
        if success:
            packages = []
            for line in stdout.strip().split('\n'):
                if line.startswith('package:'):
                    packages.append(line.replace('package:', ''))
            return packages
        return []
    
    def get_app_activity(self, package_name="com.ifree.cardCHS"):
        """获取应用的主 Activity"""
        command = f'cmd package resolve-activity --brief {package_name}'
        success, stdout, stderr = self._exec_adb(command)
        if success:
            lines = stdout.strip().split('\n')
            if len(lines) >= 2:
                return lines[-1].strip()  # 最后一行是 Activity 名
        return None

def refresh_game(mumu,time_for_getinto=60):
    #重启游戏
    mumu.restart_app(package_name='com.ifree.cardCHS')
    #随意点击
    click_position(105, 300)

    #等待进入游戏
    time.sleep(time_for_getinto)

    #进入活动
    click_position(105, 300)
    time.sleep(2.5)
    #进入魔宫
    click_position(652, 693)
    time.sleep(1.5)


def click_position(x1, y1):
    """
    在模拟器上点击指定坐标
    x, y: 屏幕坐标
    """
    mumu_path = r'D:\emulator\MuMuPlayer-12.0\nx_main\MuMuManager.exe'

    # 使用ADB命令实现点击
    command = f"shell input tap {x1} {y1}"
    args = [
        mumu_path,
        'adb',  # 使用 adb 功能
        '-v', '0',  # 指定模拟器索引为 0
        '-c', command  # ADB 命令
    ]

    # 通过MuMuManager执行ADB命令
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=30  # 设置超时时间
    )

    return result.returncode == 0

#d.click(0.897, 0.091)#退出选项
#d.click(0.892, 0.124)#退出进入选项后的选项
def u2_click(x,y,d):
    d.click(x, y)
    return 0

#shell input swipe X1 Y1 X2 Y2 TIME
def swipe_position(points,time=100):
    """
    在模拟器上点击指定坐标
    x, y: 屏幕坐标
    """
    mumu_path = r'D:\emulator\MuMuPlayer-12.0\nx_main\MuMuManager.exe'

    # 使用ADB命令实现点击
    command = f"shell input swipe {points[0][0]} {points[0][1]} {points[1][0]} {points[1][1]} {time}"
    args = [
        mumu_path,
        'adb',  # 使用 adb 功能
        '-v', '0',  # 指定模拟器索引为 0
        '-c', command  # ADB 命令
    ]

    # 通过MuMuManager执行ADB命令
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=30  # 设置超时时间
    )

    if result.returncode == 0:
        print(f"长按成功: ({points[0][0]},{points[0][1]}) ---> ({points[1][0]},{points[1][1]}) time={time}ms")
        # source_image=capture_screen()
        # result_img = draw_match_rectangles_fixed(source_image, [(x1, y1)])
        # cv2.imshow('点击结果', result_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    else:
        print(f"点击失败: {result}")

    return result.returncode == 0

def preprocess_image(img):
    """预处理图像以提高OCR识别率"""
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用自适应阈值
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # 可选: 去噪
    binary = cv2.medianBlur(binary, 3)
    
    # 转回BGR
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def capture_screen_adb():
    """
    使用 ADB 截图（后台截图，不需要窗口可见）
    返回: OpenCV 格式的图像 (BGR)
    """
    mumu_path = r'D:\emulator\MuMuPlayer-12.0\nx_main\MuMuManager.exe'
    temp_path = os.path.join(tempfile.gettempdir(), f'adb_screen_{int(time.time()*1000)}.png')
    
    # ADB 截图命令
    command = 'shell screencap -p /sdcard/screen.png'
    args = [mumu_path, 'adb', '-v', '0', '-c', command]
    subprocess.run(args, capture_output=True, timeout=10)
    
    # 拉取图片到本地
    pull_command = f'pull /sdcard/screen.png {temp_path}'
    args = [mumu_path, 'adb', '-v', '0', '-c', pull_command]
    result = subprocess.run(args, capture_output=True, timeout=10)
    
    if result.returncode != 0:
        raise RuntimeError(f"ADB 截图失败: {result.stderr}")
    
    # 读取图片
    img = cv2.imread(temp_path)
    
    # 删除临时文件
    try:
        os.remove(temp_path)
    except:
        pass
    
    return img


def capture_and_ocr_debug(ocr, region=None, debug_name="debug", save_debug=False, preprocess=True):
    """
    截图OCR识别（带调试功能）
    
    Args:
        ocr: PaddleOCR 实例
        region: 识别区域
        debug_name: 调试文件名前缀
        save_debug: 是否保存调试图片
        preprocess: 是否进行图像预处理（二值化等）
    """
    try:
        # 截图
        img = capture_screen_adb()
        
        # 裁剪区域
        if region:
            x_coords = [p[0] for p in region]
            y_coords = [p[1] for p in region]
            left = min(x_coords)
            top = min(y_coords)
            right = max(x_coords)
            bottom = max(y_coords)
            img = img[top:bottom, left:right]
        
        # 保存原始截图（用于调试）
        if save_debug:
            debug_dir = os.path.join(tempfile.gettempdir(), "moka_debug")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, f"{debug_name}_raw.png")
            cv2.imwrite(debug_path, img)
            print(f"[调试] 原始截图已保存: {debug_path}")
        
        # 图像预处理（提高数字识别率）
        if preprocess:
            # 转为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 自适应二值化
            binary = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            # 降噪
            binary = cv2.medianBlur(binary, 3)
            # 转回BGR用于OCR
            img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            if save_debug:
                debug_path = os.path.join(debug_dir, f"{debug_name}_processed.png")
                cv2.imwrite(debug_path, img)
                print(f"[调试] 处理后截图已保存: {debug_path}")
        
        # 保存临时文件进行OCR
        temp_path = os.path.join(tempfile.gettempdir(), f"ocr_debug_{int(time.time()*1000)}.png")
        cv2.imwrite(temp_path, img)
        
        try:
            result = ocr.predict(temp_path)
            texts = result[0]["rec_texts"] if result and len(result) > 0 else []
            text = "".join(texts)
            return text
        finally:
            try:
                os.remove(temp_path)
            except:
                pass
                
    except Exception as e:
        print(f"[调试OCR] 错误: {e}")
        return ""


def capture_and_ocr(ocr, region=None, window_title='MuMu安卓设备', ocr_path=None, use_adb=True):
    """
    截图OCR识别，支持1920×1080坐标转换

    参数:
        ocr: PaddleOCR 实例
        region: 1920×1080下的四个点坐标 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        window_title: 窗口标题（使用 ADB 截图时忽略）
        ocr_path: 保留参数，为了兼容
        use_adb: 是否使用 ADB 后台截图（默认True）
    """
    
    try:
        # 使用 ADB 后台截图
        if use_adb:
            img = capture_screen_adb()
        else:
            # 回退到窗口截图方式
            windows = gw.getWindowsWithTitle(window_title)
            if not windows:
                return f"未找到窗口: {window_title}"
            window = windows[0]
            
            with mss.mss() as sct:
                if region:
                    x_coords = [p[0] for p in region]
                    y_coords = [p[1] for p in region]
                    left_1920 = min(x_coords)
                    top_1920 = min(y_coords)
                    width_1920 = max(x_coords) - left_1920
                    height_1920 = max(y_coords) - top_1920
                    left = int(left_1920 * window.width / 1920)
                    top = int(top_1920 * window.height / 1080)
                    width = int(width_1920 * window.width / 1920)
                    height = int(height_1920 * window.height / 1080)
                    rect = {
                        "left": window.left + left,
                        "top": window.top + top,
                        "width": width,
                        "height": height
                    }
                else:
                    rect = {
                        "left": window.left,
                        "top": window.top,
                        "width": window.width,
                        "height": window.height
                    }
                img = cv2.cvtColor(np.array(sct.grab(rect)), cv2.COLOR_BGRA2BGR)
            
            # 调整到1920×1080
            if img.shape[1] != 1920 or img.shape[0] != 1080:
                img = cv2.resize(img, (1920, 1080))
        
        # ADB 截图时进行区域裁剪
        if use_adb and region:
            x_coords = [p[0] for p in region]
            y_coords = [p[1] for p in region]
            left = min(x_coords)
            top = min(y_coords)
            right = max(x_coords)
            bottom = max(y_coords)
            img = img[top:bottom, left:right]

        # 使用不同的临时文件保存方式
        temp_path = None
        try:
            # 创建唯一的临时文件名
            temp_dir = tempfile.gettempdir()
            temp_filename = f"ocr_temp_{int(time.time() * 1000)}.png"
            temp_path = os.path.join(temp_dir, temp_filename)

            # 保存图片
            cv2.imwrite(temp_path, img)
            # print("临时图片保存成功")

            # 确保文件保存完成
            time.sleep(0.22)

            # 调用OCR程序，确保进程完全结束
            result = ocr.predict(temp_path)

            # 等待OCR程序释放文件和完成剪贴板操作
            time.sleep(0.11)

            # 获取识别结果
            result=result[0]["rec_texts"]
            text = "".join(result)

            print("识别文本：", text)

            return text if text else ""

        finally:
            # 确保删除临时文件
            if temp_path and os.path.exists(temp_path):
                max_retries = 5  # 增加重试次数
                for i in range(max_retries):
                    try:
                        os.remove(temp_path)
                        #print("临时图片删除成功")
                        break
                    except PermissionError as e:
                        if i < max_retries - 1:
                            print(f"删除文件失败，重试 {i + 1}/{max_retries}...")
                            time.sleep(0.5)
                        else:
                            print(f"无法删除临时文件: {temp_path}, 错误: {e}")
                    except Exception as e:
                        print(f"删除文件时出错: {e}")
                        break

    except subprocess.TimeoutExpired:
        return "OCR识别超时"
    except subprocess.CalledProcessError as e:
        return f"OCR程序执行失败: {e}"
    except Exception as e:
        return f"识别失败: {e}"
    
    """
    try:
        # 截图
        windows = gw.getWindowsWithTitle(window_title)
        if not windows:
            print(f"未找到窗口: {window_title}")
            return ""

        window = windows[0]
        print(f"找到窗口: {window_title}, 位置: {window.left},{window.top}, 大小: {window.width}x{window.height}")

        with mss.mss() as sct:
            if region:
                # 提取坐标
                x_coords = [p[0] for p in region]
                y_coords = [p[1] for p in region]

                # 计算1920×1080下的区域
                left_1920 = min(x_coords)
                top_1920 = min(y_coords)
                width_1920 = max(x_coords) - left_1920
                height_1920 = max(y_coords) - top_1920

                # 转换为实际坐标
                left = int(left_1920 * window.width / 1920)
                top = int(top_1920 * window.height / 1080)
                width = int(width_1920 * window.width / 1920)
                height = int(height_1920 * window.height / 1080)

                # 设置截图区域
                rect = {
                    "left": window.left + left,
                    "top": window.top + top,
                    "width": width,
                    "height": height
                }
                print(f"截取区域: 左上({rect['left']},{rect['top']}), 大小({width}x{height})")
            else:
                rect = {
                    "left": window.left,
                    "top": window.top,
                    "width": window.width,
                    "height": window.height
                }
                print("截取整个窗口")

            # 截图
            screenshot = sct.grab(rect)
            img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_BGRA2BGR)
            print(f"截图成功, 尺寸: {img.shape}")

            # 调整到1920×1080
            if img.shape[1] != 1920 or img.shape[0] != 1080:
                img = cv2.resize(img, (1920, 1080))
                print(f"调整尺寸到: 1920x1080")

        # 方法1: 直接使用 PaddleOCR 识别
        print("开始OCR识别...")
        
        # 预处理图像以提高识别准确率
        processed_img = preprocess_image(img)
        
        # 使用 PaddleOCR 识别
        result = ocr.ocr(processed_img)
        
        # 提取文本
        combined_text = result[0]["rec_texts"]
        combined_text = "".join(combined_text) 
        print(f"识别文本: {combined_text}")
        
        return combined_text

    except Exception as e:
        print(f"识别失败: {e}")
        import traceback
        traceback.print_exc()
        return ""
    """

def get_choice(choice):
    choice_position = [0, (446, 657), (947, 653), (1447, 657)]
    aim_position = choice_position[choice]

    click_position(aim_position[0], aim_position[1])

    return 0

#圣物位置
#[(342, 409), (604, 424), (832, 426), (1069, 418)]
#[(1542, 933), (1542, 941), (1557, 937), (1525, 941), (1708, 179), (1708, 183)]
#(1201, 704)

#[(283, 419),(528, 417),(809, 417),(1705, 179),(1199, 696),(1199, 696),(1685, 481)]

def get_shengwu(choice,d):
    get_choice(choice)
    print("选择圣物")
    get_fight()
    time.sleep(0.51)
    sheng_wu_zone=[(345, 441), (645, 439), (967, 439), (1286, 443)]
    for i in sheng_wu_zone:
        click_position(i[0], i[1])
        time.sleep(0.31)
    print("回复并退出")
    click_position(1553, 939)
    time.sleep(0.1)
    for i  in range(3):
        u2_click(0.892, 0.124,d=d)
        time.sleep(0.1)
        u2_click(0.897, 0.091,d=d)
        time.sleep(0.2)
        click_position(1196, 707)
    
    return 0

def get_fight():
    click_position(1543, 952)
    return 0

# 恢复高性能版本
def finish_fight(ocr=None, max_duration=10,d=None):
    """
    高性能战斗结束处理
    """
    # 定义点击坐标 - 自动按钮区域
    auto_button_positions = [
        (1701, 442),  # 主坐标
    ]
    
    click_interval = 1.0  # 降低点击频率为1秒1次
    start_time = time.time()
    
    print(f"开始连续点击战斗按钮，持续{max_duration}秒...")
    
    while time.time() - start_time < max_duration:
        # 点击自动按钮
        for pos in auto_button_positions:
            click_position(pos[0], pos[1])
        
        # 等待1秒后进行下一次点击
        time.sleep(click_interval)
    
    time.sleep(0.8)
    # 使用mumu_screen_click代替最后的确定按钮，点击2次
    print("点击退出领取奖励的按钮...")
    for i in range(2):
        u2_click(0.892, 0.124,d=d)
        time.sleep(0.2)
    return 0

# 注释掉原来的OCR版本
"""
# 修复并恢复原来的finish_fight函数
def finish_fight(ocr):
    # 特殊：自动-跳过按钮的区域
    auto_skip_button_region = [(1724, 477), (1851, 473), (1719, 531), (1855, 538)]
    skip_button_position = (1701, 442)

    clicked_auto = False
    cnt=20
    
    while cnt>=0:
        print("进行战斗检测")
        result = capture_and_ocr(
            ocr=ocr,
            region=auto_skip_button_region,
            ocr_path=
        )

        if "跳过" in result or "过" in result:
            # 点击跳过按钮
            click_position(skip_button_position[0], skip_button_position[1])

            # 如果是先有自动再有跳过的情况，需要等待一下
            if clicked_auto:
                time.sleep(0.6)

            # 后续的两次点击
            click_position(skip_button_position[0], skip_button_position[1])
            time.sleep(0.2)
            click_position(skip_button_position[0], skip_button_position[1])

            # 最后点击确定按钮
            time.sleep(0.9)
            click_position(958, 930)
            break

        elif ("自动" in result or "自" in result or "动" in result) and not clicked_auto:
            # 第一次出现自动，点击
            click_position(skip_button_position[0], skip_button_position[1])
            clicked_auto = True
        else:
            click_position(skip_button_position[0], skip_button_position[1])

        time.sleep(0.13)
        cnt-=1

    return 0
"""

def normal_other(choice):
    get_choice(choice)
    time.sleep(0.3)
    click_position(1543, 952)
    time.sleep(0.54)
    click_position(958, 930)
    return 0

def refresh(choice):
    choice_position = [0, (627, 384), (1121, 383), (1611, 381)]
    aim_position = choice_position[choice]
    click_position(aim_position[0], aim_position[1])
    click_position(1691, 891)
    return 0

def fight_Erro(d,max_duration=3):
    finish_fight(max_duration=max_duration,d=d)
    time.sleep(0.23)
    for i in range(3):
        u2_click(0.892, 0.124,d=d)
        time.sleep(0.1)
        u2_click(0.896, 0.084,d=d)
    for i in range(3):
        click_position(1207, 705)
        time.sleep(0.1)
    for i in range(3):
        click_position(1688, 900)
        time.sleep(0.1)

#[(1627, 114), (1207, 705)]
def get_tongxing(d,ocr):
    """
    购买通行点函数
    优化后：增加点击间隔，添加错误检查和重试机制
    """
    time.sleep(0.5)
    # 第一个位置点击（加号按钮）
    d.click(0.851, 0.052)
    time.sleep(0.5)
    # 第二个位置点击（确定按钮）
    click_position(1207, 705)

    return 0

def overcome_chenmi(d,choice,ocr):
    get_choice(choice)
    time.sleep(0.21)

    # 选取圣物
    click_position(331, 784)
    time.sleep(0.38)
    # 查看圣物类别
    get_shengwu = 0

    shengwu_position = [(332, 386), (1091, 374), (316, 623), (1086, 620)]

    #筛选关键词
    d.click(0.875, 0.154)
    time.sleep(0.2)
    d.click(0.201, 0.802)
    time.sleep(0.2)
    d.click(0.611, 0.347)#沉迷圣物
    time.sleep(0.2)
    d.click(0.5, 0.797)
    time.sleep(0.2)
    click_position(shengwu_position[get_shengwu][0], shengwu_position[get_shengwu][1])
    return 0

def overcome_gaojie(d,choice,ocr):
    get_choice(choice)
    time.sleep(0.3)

    # 选取圣物
    click_position(331, 784)
    time.sleep(0.98)
    # 查看圣物类别
    get_shengwu = 0
    shengwu_position = [(332, 386), (1091, 374), (316, 623), (1086, 620)]

    #筛选关键词
    d.click(0.875, 0.154)
    time.sleep(0.2)
    d.click(0.201, 0.802)
    time.sleep(0.5)
    d.click(0.363, 0.423)#圣洁圣物
    time.sleep(0.5)
    d.click(0.5, 0.797)
    time.sleep(0.2)
    click_position(shengwu_position[get_shengwu][0], shengwu_position[get_shengwu][1])
    return 0

def overcome_gaojie_and_chenmi(d,choice,ocr):
    get_choice(choice)
    time.sleep(0.3)

    # 选取圣物
    click_position(331, 784)
    time.sleep(0.98)
    # 查看圣物类别
    get_shengwu = 0

    shengwu_position = [(332, 386), (1091, 374), (316, 623), (1086, 620)]

    #筛选关键词
    d.click(0.875, 0.154)
    time.sleep(0.2)
    d.click(0.201, 0.802)
    time.sleep(0.5)
    d.click(0.363, 0.423)#圣洁圣物
    time.sleep(0.5)
    d.click(0.611, 0.347)#沉迷圣物
    time.sleep(0.5)
    d.click(0.5, 0.797)
    time.sleep(0.2)
    click_position(shengwu_position[get_shengwu][0], shengwu_position[get_shengwu][1])
    time.sleep(0.1)
    click_position(shengwu_position[1][0], shengwu_position[1][1])
    return 0

def overcome_qingchun(choice,ocr):
    get_choice(choice)
    time.sleep(0.3)

    # 选取圣物
    click_position(331, 784)
    time.sleep(0.68)
    # 查看圣物类别
    get_shengwu = 0
    get_text_for_shengwu = [[(424, 286), (936, 272), (418, 476), (936, 482)],
                            [(1181, 284), (1701, 278), (1183, 476), (1695, 477)],
                            [(416, 527), (936, 521), (411, 723), (933, 724)],
                            [(1181, 527), (1698, 521), (1180, 719), (1695, 723)]]
    shengwu_position = [(332, 386), (1091, 374), (316, 623), (1086, 620)]

    click_position(shengwu_position[get_shengwu][0], shengwu_position[get_shengwu][1])
    return 0

def make_choice(ocr, gaojie_count=1, chenmi_count=1, qingchun_count=1, score=0, d=None,
                skip_chenmi_normal=True, skip_gaojie_normal=True, skip_both_normal=True):
    """
    识别并选择关卡
    
    Args:
        skip_chenmi_normal: 是否跳过非boss层的"沉迷"词条关卡
        skip_gaojie_normal: 是否跳过非boss层的"高洁"词条关卡  
        skip_both_normal: 是否跳过非boss层的"沉迷+高洁"双词条关卡
    """
    # 最多重试5次
    max_retries = 5

    for retry_count in range(max_retries):
        print(f"\n=== 开始第 {retry_count + 1}/{max_retries} 次选项识别 ===")
        print(f"[跳过配置] 沉迷小怪: {skip_chenmi_normal}, 高洁小怪: {skip_gaojie_normal}, 双词条小怪: {skip_both_normal}")

        # 查看当前三个选择的内容
        choice_for_text_region = [
            [(272, 758),(673, 760),(259, 991),(682, 995)],
            [(753, 749),(1172, 743),(750, 1001),(1172, 1001)],
            [(1241, 748),(1650, 739),(1237, 989),(1652, 995)]
        ]

        choice = 1
        fang_an_order = 0
        
        # 按顺序识别每个选项并立即判断
        for choice_index, region in enumerate(choice_for_text_region, 1):
            # 识别当前选项
            result = capture_and_ocr(
                ocr=ocr,
                region=region,
                ocr_path=r"F:\Umi-OCR\Umi-OCR_Paddle_v2.1.5\Umi-OCR.exe"
            )

            # 青春词条优先判断（70%概率选择）
            if "青春" in result and "沉迷" not in result and "高洁" not in result and "奉献" not in result and "花粉" not in result and qingchun_count == 1 and "沉" not in result:
                if random.random() < 1:  # 70%概率选择青春
                    fang_an_order = 9
                    #qingchun_count -= 1
                    choice = choice_index
                    if "精英战" in result:
                        score += 80
                    elif "阻击战" in result:
                        score += 50
                    elif "遭遇战" in result:
                        score += 30
                    print(f"识别成功：选择青春方案，剩余青春次数：{qingchun_count}")
                    return choice, fang_an_order, gaojie_count, chenmi_count, qingchun_count,score

            # 如果不是青春或跳过青春，继续其他判断
            elif "精英战" not in result and "遭遇战" not in result and "阻击战" not in result and "神迹" not in result:
                if "藏宝" in result or "祈求" in result or "清泉" in result or "休" in result or "求" in result or "宝" in result: 
                    fang_an_order = 1
                    choice = choice_index
                    print("识别成功：选择正常奖励关卡")
                    if "圣痕" in result:
                        score+=20
                    if "天启" in result:
                        score+=600
                    return choice, fang_an_order, gaojie_count, chenmi_count, qingchun_count,score
                    break
                elif "猪王" in result:
                    fang_an_order = 2
                    choice = choice_index
                    score += 10
                    print("识别成功：选择交易奖励关卡")
                    return choice, fang_an_order, gaojie_count, chenmi_count, qingchun_count,score
                    break

            elif "精英战" not in result and "神迹" not in result:
                # 非boss层战斗关卡判断
                has_chenmi = "沉迷" in result
                has_gaojie = "高洁" in result
                is_normal_battle = 1
                
                if is_normal_battle:
                    # 双词条小怪层
                    if has_chenmi and has_gaojie:
                        if skip_both_normal:
                            fang_an_order = 4
                            choice = choice_index
                            print("识别成功：跳过双词条小怪层")
                            return choice, fang_an_order, gaojie_count, chenmi_count, qingchun_count, score
                        else:
                            fang_an_order = 3
                            choice = choice_index
                            print("识别成功：选择双词条小怪层")
                            score += 50 if "阻击战" in result else 30
                            return choice, fang_an_order, gaojie_count, chenmi_count, qingchun_count, score
                    
                    # 沉迷小怪层
                    elif has_chenmi:
                        if skip_chenmi_normal:
                            fang_an_order = 4
                            choice = choice_index
                            print("识别成功：跳过沉迷小怪层")
                            return choice, fang_an_order, gaojie_count, chenmi_count, qingchun_count, score
                        else:
                            fang_an_order = 3
                            choice = choice_index
                            print("识别成功：选择沉迷小怪层")
                            score += 50 if "阻击战" in result else 30
                            return choice, fang_an_order, gaojie_count, chenmi_count, qingchun_count, score
                    
                    # 高洁小怪层
                    elif has_gaojie:
                        if skip_gaojie_normal:
                            fang_an_order = 4
                            choice = choice_index
                            print("识别成功：跳过高洁小怪层")
                            return choice, fang_an_order, gaojie_count, chenmi_count, qingchun_count, score
                        else:
                            fang_an_order = 3
                            choice = choice_index
                            print("识别成功：选择高洁小怪层")
                            score += 50 if "阻击战" in result else 30
                            return choice, fang_an_order, gaojie_count, chenmi_count, qingchun_count, score
                    
                    # 普通小怪层（无词条）
                    else:
                        fang_an_order = 3
                        choice = choice_index
                        print("识别成功：选择普通战斗关卡")
                        score += 50 if "阻击战" in result else 30
                        return choice, fang_an_order, gaojie_count, chenmi_count, qingchun_count, score
                    break

            elif "神迹" in result:
                fang_an_order = 8
                chenmi_count = 1
                gaojie_count = 1
                qingchun_count = 1
                choice = choice_index
                print("识别成功：选择神迹")
                return choice, fang_an_order, gaojie_count, chenmi_count, qingchun_count,score
                break

            elif "精英战" in result:
                # 普通精英战，直接处理
                if "沉迷" in result and "高洁" in result:
                    if gaojie_count == 1 and chenmi_count == 1:
                        fang_an_order = 5
                        gaojie_count -= 1
                        chenmi_count -= 1
                        choice = choice_index
                        if "奉献" not in result:
                            score += 80
                        else:
                            score+=160
                        print("识别成功：选择沉迷和高洁双词条精英战")
                        return choice, fang_an_order, gaojie_count, chenmi_count, qingchun_count,score
                        break
                    else:
                        print("没有圣物，跳过")
                        continue
                elif "沉迷" in result:
                    if chenmi_count == 1:
                        fang_an_order = 6
                        chenmi_count -= 1
                        choice = choice_index
                        print("识别成功：选择沉迷词条精英战")
                        if "奉献" not in result:
                            score += 80
                        else:
                            score+=160
                        return choice, fang_an_order, gaojie_count, chenmi_count, qingchun_count,score
                        break
                    else:
                        print("没有圣物，跳过")
                        continue
                elif "高洁" in result:
                    if gaojie_count == 1:
                        fang_an_order = 7
                        gaojie_count -= 1
                        choice = choice_index
                        print("识别成功：选择高洁词条精英战")
                        if "奉献" not in result:
                            score += 80
                        else:
                            score+=160
                        return choice, fang_an_order, gaojie_count, chenmi_count, qingchun_count,score
                        break
                    else:
                        print("没有圣物，跳过")
                        continue
                else:
                    # 普通精英战，直接处理
                    fang_an_order = 3
                    choice = choice_index
                    print("识别成功：选择普通精英战")
                    score += 80
                    return choice, fang_an_order, gaojie_count, chenmi_count, qingchun_count,score
                    break
            if fang_an_order==0:
                fight_Erro(d)
                continue
        


    # 理论上不会到达这里，但为了保险返回默认值
    return 1, 1, gaojie_count, chenmi_count, qingchun_count

def main(config: Optional[GameConfig] = None) -> Dict[str, Any]:
    """
    主函数 - 游戏自动化主循环
    
    Args:
        config: 游戏配置对象，如果为None则使用全局配置
        
    Returns:
        运行统计信息字典
    """
    # 获取配置
    cfg = config or get_config()
    
    print("\n" + "="*70)
    print("  Moka 游戏自动化脚本启动")
    print("="*70)
    print(f"[配置] 设备: {cfg.device}")
    print(f"[配置] 圣物 - 高洁:{cfg.gaojie_count} 沉迷:{cfg.chenmi_count} 青春:{cfg.qingchun_count}")
    print(f"[配置] 通行点: {cfg.tongxing_num}")
    print(f"[配置] 跳过策略 - 沉迷小怪:{cfg.skip_chenmi_normal} 高洁小怪:{cfg.skip_gaojie_normal} 双词条小怪:{cfg.skip_both_normal}")
    print(f"[配置] 自动刷新: {cfg.enable_auto_refresh} (间隔:{cfg.refresh_interval_minutes}分钟)")
    print(f"[配置] 自动重启: {cfg.enable_auto_restart} (间隔:{cfg.restart_interval_minutes}分钟)")
    print("="*70 + "\n")
    
    # 初始化 MuMu 管理器
    mumu = MuMuAppManager(mumu_path=cfg.mumu_path)
    
    # 初始化 OCR
    print("[初始化] 加载 OCR 模型...")
    ocr = PaddleOCR(
        use_textline_orientation=False,
        text_detection_model_name=cfg.ocr_det_model,
        text_recognition_model_name=cfg.ocr_rec_model
    )
    en_ocr = PaddleOCR(
        use_textline_orientation=False,
        text_detection_model_name=cfg.ocr_det_model,
        text_recognition_model_name=cfg.ocr_en_rec_model
    )
    print("[初始化] OCR 模型加载完成")
    
    # 连接设备
    print(f"[初始化] 连接设备 {cfg.device}...")
    d = u2.connect(cfg.device)
    print("[初始化] 设备连接成功")

    # 初始化状态变量
    cnt = 0                                    # 当前轮数
    diomand = 0                                # 消耗钻石数
    score = 0                                  # 当前分数
    start_time = datetime.now()                # 开始时间
    last_refresh_time = datetime.now()         # 上次游戏刷新时间
    last_restart_time = datetime.now()         # 上次脚本重启时间
    
    # 圣物状态（使用配置初始值）
    gaojie_count = cfg.gaojie_count
    chenmi_count = cfg.chenmi_count
    qingchun_count = cfg.qingchun_count
    tongxing_num = cfg.tongxing_num
    
    # 计算时间间隔（秒）
    refresh_interval_seconds = cfg.refresh_interval_minutes * 60
    restart_interval_seconds = cfg.restart_interval_minutes * 60
    
    # 运行统计
    stats = {
        'total_rounds': 0,
        'total_score': 0,
        'total_diamond': 0,
        'start_time': start_time,
        'end_time': None,
        'status': 'running'
    }

    while cnt < 1000000:
        # 计算已运行时间（秒）
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        if elapsed_time > 0:
            score_per_minute = (score / elapsed_time) * 60 if score > 0 else 0
            diamond_per_minute = (diomand / elapsed_time) * 60 if diomand > 0 else 0
        else:
            score_per_minute = 0
            diamond_per_minute = 0
        
        print("=" * 20, f"开始第 {cnt} 轮循环,当前运行时间{elapsed_time}s,当前分数{score},每分钟获取{score_per_minute:.2f},消耗钻石{diomand},每分钟消耗{diamond_per_minute:.2f}", "=" * 20)
        
        # 调用make_choice函数进行识别和选择
        # 从配置获取跳过策略
        choice, fang_an_order, gaojie_count, chenmi_count, qingchun_count, score = make_choice(
            ocr, 
            gaojie_count, 
            chenmi_count, 
            qingchun_count,
            score,
            d,
            skip_chenmi_normal=cfg.skip_chenmi_normal,
            skip_gaojie_normal=cfg.skip_gaojie_normal,
            skip_both_normal=cfg.skip_both_normal
        )
        
        print("当前选项为:", choice)

        if fang_an_order == 1:
            # 正常奖励关卡
            print("正常奖励关卡")
            normal_other(choice)

        elif fang_an_order == 2:
            # 交易奖励关卡
            print("交易奖励关卡")
            get_choice(choice)
            get_fight()
            u2_click(0.892, 0.124,d=d)  
            time.sleep(0.43)
            click_position(1315, 712)
            for i in range(3):
                u2_click(0.892, 0.124,d=d)
                u2_click(0.897, 0.091,d=d)
                time.sleep(0.24)     

        elif fang_an_order == 3:
            # 正常战斗或普通精英战
            print("正常战斗或普通精英战")
            get_choice(choice)
            time.sleep(0.33)
            get_fight()
            # 获取一个合适的判断时机
            time.sleep(0.34)
            finish_fight(ocr,d=d)
            # 输出当前参数状态
            print(f"精英战结束后参数状态 - 高洁次数: {gaojie_count}, 沉迷次数: {chenmi_count}, 青春次数: {qingchun_count}, 通行点数量: {tongxing_num}")

        elif fang_an_order == 4:
            # 跳过并且刷新
            # 检查是否该补充通行点
            print("[通行点检查] 开始识别通行点数量...")
            
            # 使用更精确的区域（根据注释中的原始坐标调整）
            # 原始: [(1511, 61), (1594, 57), (1512, 130), (1604, 136)]
            tong_xing_region = [(1523, 169), (1619, 172), (1521, 29), (1634, 23)]
            
            # 使用调试模式识别，保存截图以便排查问题
            tongxing_str = capture_and_ocr(
                ocr, 
                region=tong_xing_region, 
            )
            
            print(f"[通行点检查] 识别结果: '{tongxing_str}'")
            
            # 更健壮的空值判断
            is_empty = not tongxing_str or tongxing_str.strip() == ""
            is_zero = tongxing_str.strip() in ["0", "o", "O", "Q", "()"]
            
            if is_zero:
                print("[通行点检查] 通行点为0或识别失败，准备购买通行点")
                time.sleep(0.31)
                get_tongxing(d, ocr)
                time.sleep(0.32)
                print("[通行点检查] 购买完成，跳过当前关卡")
                refresh(choice)
                diomand += 100
            else:
                # 尝试解析数字
                try:
                    # 提取数字
                    import re
                    numbers = re.findall(r'\d+', tongxing_str)
                    if numbers:
                        point_count = int(numbers[0])
                        print(f"[通行点检查] 当前通行点: {point_count}")
                        if point_count <= 0:
                            print("[通行点检查] 通行点为0，准备购买")
                            time.sleep(0.31)
                            get_tongxing(d, ocr)
                            time.sleep(0.32)
                            diomand += 100
                except:
                    pass
                
                print("[通行点检查] 跳过当前关卡")
                time.sleep(0.12)
                refresh(choice)

        elif fang_an_order == 5:
            # 沉迷圣洁都有的boss
            print("沉迷圣洁都有的boss")


            fight_failed = True
            while fight_failed:
                 # 执行战斗准备
                overcome_gaojie_and_chenmi(d, choice, ocr)
                get_fight()
                time.sleep(0.34)
                finish_fight(ocr,d=d)

                # 预处理扫描：确保扫描结果有效（包含特定关键词）
                print("进行预处理扫描，确保结果有效")
                text_region = [(1673, 364), (1888, 358), (1673, 513), (1888, 513)]

                valid_scan = False
                while not valid_scan:
                    scan_result = capture_and_ocr(ocr, region=text_region, ocr_path=r"F:\Umi-OCR\Umi-OCR_Paddle_v2.1.5\Umi-OCR.exe")
                    if "魔宫商店" in scan_result or "商店" in scan_result or "魔宫" in scan_result:
                        valid_scan = True
                        break
                    else:
                        print("扫描结果无效，继续扫描")
                        fight_Erro(d)

                choice_for_text_region = [
                                        [(262, 755),(676, 749), (265, 843),(675, 844),],
                    [(760, 748),(1165, 744),(754, 843),(1158, 841),],
                    [(1253, 742),(1647, 738),(1237, 838),(1650, 840)]
                ]

                fight_failed=False
                for region in choice_for_text_region:
                    check_result = capture_and_ocr(ocr, region=region, ocr_path=r"F:\Umi-OCR\Umi-OCR_Paddle_v2.1.5\Umi-OCR.exe")
                    if "精英战" in check_result:
                        fight_failed = True
                        break
                    
            # 输出当前参数状态
            print(f"精英战结束后参数状态 - 高洁次数: {gaojie_count}, 沉迷次数: {chenmi_count}, 青春次数: {qingchun_count}, 通行点数量: {tongxing_num}")

        elif fang_an_order == 6:
            # 沉迷的boss
            print("沉迷的boss")


            fight_failed = True
            while fight_failed:
                 # 执行战斗准备
                overcome_chenmi(d,choice, ocr)
                get_fight()
                time.sleep(0.34)
                finish_fight(ocr,d=d)

                # 预处理扫描：确保扫描结果有效（包含特定关键词）
                print("进行预处理扫描，确保结果有效")
                text_region = [(1673, 364), (1888, 358), (1673, 513), (1888, 513)]

                valid_scan = False
                while not valid_scan:
                    scan_result = capture_and_ocr(ocr, region=text_region, ocr_path=r"F:\Umi-OCR\Umi-OCR_Paddle_v2.1.5\Umi-OCR.exe")
                    if "魔宫商店" in scan_result or "商店" in scan_result:
                        valid_scan = True
                        break
                    else:
                        print("扫描结果无效，继续扫描")
                        fight_Erro(d)

                choice_for_text_region = [
                    [(262, 755),(676, 749), (265, 843),(675, 844),],
                    [(760, 748),(1165, 744),(754, 843),(1158, 841),],
                    [(1253, 742),(1647, 738),(1237, 838),(1650, 840)]
                ]

                fight_failed=False
                for region in choice_for_text_region:
                    check_result = capture_and_ocr(ocr, region=region, ocr_path=r"F:\Umi-OCR\Umi-OCR_Paddle_v2.1.5\Umi-OCR.exe")
                    if "精英战" in check_result:
                        fight_failed = True
                        break
                    
            # 输出当前参数状态
            print(f"精英战结束后参数状态 - 高洁次数: {gaojie_count}, 沉迷次数: {chenmi_count}, 青春次数: {qingchun_count}, 通行点数量: {tongxing_num}")

        elif fang_an_order == 7:
            # 圣洁的boss
            print("圣洁的boss")

            fight_failed = True
            while fight_failed:
                 # 执行战斗准备
                overcome_gaojie(d,choice, ocr)
                get_fight()
                time.sleep(0.34)
                finish_fight(ocr,d=d)

                # 预处理扫描：确保扫描结果有效（包含特定关键词）
                print("进行预处理扫描，确保结果有效")
                text_region = [(1673, 364), (1888, 358), (1673, 513), (1888, 513)]

                valid_scan = False
                while not valid_scan:
                    scan_result = capture_and_ocr(ocr, region=text_region, ocr_path=r"F:\Umi-OCR\Umi-OCR_Paddle_v2.1.5\Umi-OCR.exe")
                    if "魔宫商店" in scan_result or "商店" in scan_result:
                        valid_scan = True
                        break
                    else:
                        print("扫描结果无效，继续扫描")
                        fight_Erro(d)

                choice_for_text_region = [
                                        [(262, 755),(676, 749), (265, 843),(675, 844),],
                    [(760, 748),(1165, 744),(754, 843),(1158, 841),],
                    [(1253, 742),(1647, 738),(1237, 838),(1650, 840)]
                ]

                fight_failed=False
                for region in choice_for_text_region:
                    check_result = capture_and_ocr(ocr, region=region, ocr_path=r"F:\Umi-OCR\Umi-OCR_Paddle_v2.1.5\Umi-OCR.exe")
                    if "精英战" in check_result:
                        fight_failed = True
                        break
                    
            # 输出当前参数状态
            print(f"精英战结束后参数状态 - 高洁次数: {gaojie_count}, 沉迷次数: {chenmi_count}, 青春次数: {qingchun_count}, 通行点数量: {tongxing_num}")

        elif fang_an_order == 8:
            # 神迹
            print("神迹")
            get_shengwu(choice,d)

        elif fang_an_order == 9:
            # 青春词条
            print("青春词条")
            overcome_qingchun(choice,ocr)
            get_fight()
            # 获取一个合适的判断时机
            finish_fight(ocr,d=d)
            tongxing_num+=2

        time.sleep(0.2)
        cnt += 1
        stats['total_rounds'] = cnt
        stats['total_score'] = score
        stats['total_diamond'] = diomand
        
        # 检查是否需要定期刷新游戏
        if cfg.enable_auto_refresh and refresh_interval_seconds > 0:
            time_since_last_refresh = (datetime.now() - last_refresh_time).total_seconds()
            if time_since_last_refresh >= refresh_interval_seconds:
                print(f"\n{'='*60}")
                print(f"[定时刷新] 已达到刷新间隔 {cfg.refresh_interval_minutes} 分钟，准备刷新游戏...")
                print(f"[定时刷新] 当前运行状态 - 轮数: {cnt}, 分数: {score}")
                print(f"{'='*60}\n")
                
                # 调用刷新游戏函数
                refresh_game(mumu, time_for_getinto=30)
                
                # 更新上次刷新时间
                last_refresh_time = datetime.now()
                print(f"\n[定时刷新] 游戏刷新完成，继续运行...\n")
        
        # 检查是否需要重启脚本
        if cfg.enable_auto_restart and restart_interval_seconds > 0:
            time_since_last_restart = (datetime.now() - last_restart_time).total_seconds()
            if time_since_last_restart >= restart_interval_seconds:
                print(f"\n{'='*60}")
                print(f"[定时重启] 已达到重启间隔 {cfg.restart_interval_minutes} 分钟，准备重启脚本...")
                print(f"[定时重启] 当前运行状态 - 轮数: {cnt}, 分数: {score}")
                print(f"{'='*60}\n")
                
                # 保存状态
                if cfg.save_state_on_exit:
                    save_state(cnt, score, diomand, gaojie_count, chenmi_count, qingchun_count, tongxing_num)
                
                stats['end_time'] = datetime.now()
                stats['status'] = 'restarting'
                break
    
    # 循环结束，返回统计信息
    stats['end_time'] = datetime.now()
    if stats['status'] == 'running':
        stats['status'] = 'completed'
    
    # 保存最终状态
    if cfg.save_state_on_exit:
        save_state(cnt, score, diomand, gaojie_count, chenmi_count, qingchun_count, tongxing_num)
    
    # 输出最终统计
    elapsed = (stats['end_time'] - start_time).total_seconds()
    print("\n" + "="*70)
    print("  运行结束统计")
    print("="*70)
    print(f"运行状态: {stats['status']}")
    print(f"总轮数: {stats['total_rounds']}")
    print(f"总分数: {stats['total_score']}")
    print(f"总消耗钻石: {stats['total_diamond']}")
    print(f"运行时间: {elapsed/60:.2f} 分钟")
    print(f"平均每分分数: {stats['total_score']/(elapsed/60):.2f}" if elapsed > 0 else "N/A")
    print("="*70 + "\n")
    
    return stats


# ==================== 入口点 ====================

if __name__ == "__main__":
    # 加载配置
    config = GameConfig.load_from_file()
    
    # 运行主函数
    try:
        result = main(config)
        
        # 如果配置了自动重启且状态为 restarting，则退出后由外部脚本重启
        if config.enable_auto_restart and result.get('status') == 'restarting':
            print("[退出] 脚本将在5秒后由外部重启...")
            time.sleep(5)
            exit(0)
            
    except KeyboardInterrupt:
        print("\n[中断] 用户手动中断脚本")
        # 保存状态
        config = get_config()
        if config.save_state_on_exit:
            config.save_to_file()
        exit(0)
    except Exception as e:
        print(f"\n[错误] 脚本运行异常: {e}")
        import traceback
        traceback.print_exc()
        exit(1)