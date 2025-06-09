



# def init_memory(self, memory_param, num_nodes, dim_edge_feat):
#     self.memory = torch.zeros((num_nodes, memory_param['dim_out']), dtype=torch.float32)
#     self.memory_ts = torch.zeros(num_nodes, dtype=torch.float32)
#     self.mailbox = torch.zeros((num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_edge_feat), dtype=torch.float32)
#     self.mailbox_ts = torch.zeros((num_nodes, memory_param['mailbox_size']), dtype=torch.float32)

def cal(num_nodes, dim_edge_feat = 172, dim_out = 100):
    res = 0
    res += num_nodes * dim_out * 4
    res += num_nodes * 4
    res += num_nodes * (2 * dim_out + dim_edge_feat) * 4
    res += num_nodes * 4
    res = res / 1024 ** 3
    print(f"node:{num_nodes}, ef:{dim_edge_feat}, hd: {dim_out} 占{res:.2f}GB")

dims = [50, 100, 200, 300, 400, 800]
num_nodes = 1140149
for dim in dims:
    cal(num_nodes, dim_out = dim)