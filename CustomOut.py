import datetime
import os

# 全局变量用于存储日志文件句柄
log_file = None

def init_log_file(logs_folder='./logs', log_filename=None):
    global log_file

    if log_filename is None:
        log_filename = datetime.datetime.now().strftime("log %Y-%m-%d %H-%M-%S.txt")
    else:
        log_filename = datetime.datetime.now().strftime(log_filename + " log %Y-%m-%d %H-%M-%S.txt")
    
        
    log_filepath = os.path.join(logs_folder, log_filename)
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
    log_file = open(log_filepath, "w")
    return log_file

def get_current_timestamp():
    """
    Generate a timestamp string in the format of [hour:min:sec.millisecond].
    """
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")
    # 截取到毫秒（前3位数字）
    return  "[" + timestamp[:-3] + "]"

def custom_print(message):
    print(get_current_timestamp(), message)  # Print to console
    if log_file is not None:
        print(get_current_timestamp(), message, file=log_file)  # Print to file