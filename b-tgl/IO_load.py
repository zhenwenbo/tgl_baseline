import multiprocessing
import time
import random
import torch
import dgl
import os
from utils import *
from config.train_conf import *

import threading

class IO_fetch:
    def __init__(self, conn, prefetch_child_conn):
        self.conn = conn
        self.prefetch_conn = prefetch_child_conn

    def init_shared_tensor(self, tensor):
        print(f"IO子进程 init shared tensor")

    def run(self):
        while True:
            if self.conn.poll():  # 检查是否有数据
                message = self.conn.recv()
                if message == "EXIT":
                    break
                function_name, args = message
                if hasattr(self, function_name):
                    # print(f"子程序调用程序: {function_name}")
                    func = getattr(self, function_name)
                    result = func(*args)
                    # print(f"子进程result传回")
                    if (function_name == 'pre_fetch'):
                        # print(f"使用prefetch传回")
                        self.prefetch_conn.send(result)
                    else:
                        self.conn.send(result)
                else:
                    self.conn.send(f"Function {function_name} not found")

def prefetch_worker_IO(conn, prefetch_child_conn):
    prefetch = IO_fetch(conn, prefetch_child_conn)
    prefetch.run()


