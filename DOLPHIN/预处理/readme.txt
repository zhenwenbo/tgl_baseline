有了一个edges.csv文件
1.原始数据预处理.py 
2.数据特征生成.py
3.disk版本预处理.py
就可以生成直接用python train.py的一个文件夹
而目前的版本（如果跑时间区间相关的数据集）前三个都为了适配有所修改 所以新开文件
1.deal_ts_csv.py(目前LASTFM_interval,TALK_interval,STACK_interval均可以)
2.data_process.py 
3.process_data_feature.py
4.disk版本预处理.py