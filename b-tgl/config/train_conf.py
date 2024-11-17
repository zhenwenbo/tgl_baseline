import json




class GlobalConfig:

    conf_path = '/raid/guorui/workspace/dgnn/b-tgl/config/train_conf'

    

    def __init__(self):
        self.data_incre = True
        self.memory_disk = False
        self.model_eval = False
        self.node_cache = False
        self.node_reorder = False
        


        filename = f'{self.conf_path}/{self.conf}'
        # 打开并读取JSON文件
        with open(filename, 'r') as file:
            # 加载JSON内容到字典
            self.config_data = json.load(file)
        
        # 遍历字典中的所有键值对
        for key, value in self.config_data.items():
            # 使用setattr动态创建属性
            setattr(self, key, value)


