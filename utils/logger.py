import logging
import time
import os


class Logger:
    def __init__(self, mode='w', title="", verbose=True):
        # create logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)  # setting level
        formatter_fh = logging.Formatter("[%(asctime)s] %(message)s")
        formatter_ch = logging.Formatter("%(message)s")

        # create file handler
        # setting path for logfile 设置日志文件名称
        start_time = time.strftime('%y-%m-%d-%H%M', time.localtime(time.time()))
        log_path = os.path.join(os.getcwd(), 'logs', title)
        if not os.path.exists(log_path):
            os.makedirs(log_path)       
        log_name = os.path.join(log_path, start_time + '.log')
        
        fh = logging.FileHandler(log_name, mode=mode)
        fh.setLevel(logging.INFO)  # setting level for outputs in logfile
        ## define format
        fh.setFormatter(formatter_fh)
        ## add handeler to the logger
        self.logger.addHandler(fh)

        if verbose:
            # create console handeler
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(formatter_ch)
            self.logger.addHandler(ch)