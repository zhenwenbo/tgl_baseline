各文件夹及函数功能
The repository is organized as follows:

- `config/`: 各种配置文件 最重要的是/home/zwb/tgl-baseline/DOLPHIN/config/train_conf 这个文件下有训练配置 config/train_conf/mem_w_valid.json（因为这个只有一处 比较简单） 对这个修改可以对很多优化起效 其他文件（可删除）
- `preprocess/`: 我感觉这个主要是在处理empire采样（过期边机制） 但是在目前的代码里并没有真正用到empire这样机制 这个后面也合并到预处理中
- `sampler/`: cpu、gpu采样及其支持函数 untils是各种功能函数都在里面
- `test/`: 这个文件夹下 有很多的涵盖各方面的测试信息  
- `预处理/`: 有了一个edges.csv文件
            1.原始数据预处理.py 
            2.数据特征生成.py
            3.disk版本预处理.py
            就可以生成直接用python train.py的一个文件夹（目前可能存在兼容问题）
            而目前的版本（如果跑时间区间相关的数据集）前三个都为了适配有所修改 所以新开文件
            1.deal_ts_csv.py(目前LASTFM_interval和TALK_interval,STACK_interval均可以,但是彼此之间不兼容)
            2.data_process.py 
            3.process_data_feature.py
            4.disk版本预处理.py
- `setup.py`: 通过 pybind11 工具将 C++ 代码编译为 Python 可调用的扩展模块
- `train.py`: Model training script.

This structure helps organize code, configuration, and data for efficient development and experimentation.