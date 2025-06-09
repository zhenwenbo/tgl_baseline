import multiprocessing
from pre_fetch import *
import time

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    parent_conn, child_conn = multiprocessing.Pipe()
    prefetch_conn, prefetch_child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=prefetch_worker, args=(child_conn, prefetch_child_conn))
    p.start()

    share_s = time.time()
    tensor_share = torch.zeros(3000, dtype = torch.int32).cuda()
    print(f"建立share {time.time() - share_s:.5f}s")



    node_num = 3000
    edge_num = 3000
    node_feat_dim = 172
    edge_feat_dim = 172
    mem_dim = 100
    mailbox_size = 1

    part_node_map = torch.zeros(node_num, dtype = torch.int32).share_memory_()
    node_feats = torch.zeros((node_num, 172), dtype = torch.float32).share_memory_()

    part_edge_map = torch.zeros(edge_num, dtype = torch.int32).share_memory_()
    edge_feats = torch.zeros((edge_num, edge_feat_dim), dtype = torch.float32).share_memory_()

    part_memory = torch.zeros((node_num, mem_dim), dtype = torch.float32).share_memory_()
    part_memory_ts = torch.zeros(node_num, dtype = torch.float32).share_memory_()
    part_mailbox = torch.zeros((node_num, mailbox_size, 2 * mem_dim + edge_feat_dim), dtype = torch.float32).share_memory_()
    part_mailbox_ts = torch.zeros((node_num, mem_dim), dtype = torch.float32).share_memory_()

    pre_same_nodes = torch.zeros(node_num, dtype = torch.int32).share_memory_()
    cur_same_nodes = torch.zeros(node_num, dtype = torch.int32).share_memory_()

    shared_tensor = (part_node_map, node_feats, part_edge_map, edge_feats, part_memory, part_memory_ts, part_mailbox, part_mailbox_ts, pre_same_nodes, cur_same_nodes)
    
    shared_ret_len = torch.zeros(len(shared_tensor), dtype = torch.int32).share_memory_()
    share_tmp_tensor = torch.zeros(300000000).share_memory_()
    shared_tensor = (*shared_tensor,shared_ret_len, share_tmp_tensor)

    parent_conn.send(('init_share_tensor', (shared_tensor,)))

    # 主进程发送要调用的函数名和args变量
    for i in range(5):
        func_name = "function"
        args = f"Task {i}"
        

        start = time.time()
        parent_conn.send((func_name, (args,tensor_share)))
        print(f"Sent: {func_name}, {args}")
        result = parent_conn.recv()
        print(f"Received: {result}")

        print(f"传递总时间{time.time() - start:.3f}s")

        part_node_map = torch.zeros(123123, dtype = torch.int32).share_memory_()

    # 发送退出信号
    parent_conn.send("EXIT")
    p.join()


