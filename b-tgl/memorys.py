import torch
import dgl
from layers import TimeEncode
from torch_scatter import scatter

class MailBox():

    def __init__(self, memory_param, num_nodes, dim_edge_feat, prefetch_conn = None, feat_buffer = None, _node_memory=None, _node_memory_ts=None,_mailbox=None, _mailbox_ts=None, _next_mail_pos=None, _update_mail_pos=None):
        self.memory_param = memory_param
        self.dim_edge_feat = dim_edge_feat
        if memory_param['type'] != 'node':
            raise NotImplementedError
        
        self.prefetch_conn = prefetch_conn
        if (prefetch_conn):
            prefetch_conn.send(('init_memory', (memory_param, num_nodes, dim_edge_feat)))
            prefetch_conn.recv()
        else:
            self.node_memory = torch.zeros((num_nodes, memory_param['dim_out']), dtype=torch.float32) if _node_memory is None else _node_memory
            self.node_memory_ts = torch.zeros(num_nodes, dtype=torch.float32) if _node_memory_ts is None else _node_memory_ts
            self.mailbox = torch.zeros((num_nodes, memory_param['mailbox_size'], 2 * memory_param['dim_out'] + dim_edge_feat), dtype=torch.float32) if _mailbox is None else _mailbox
            self.mailbox_ts = torch.zeros((num_nodes, memory_param['mailbox_size']), dtype=torch.float32) if _mailbox_ts is None else _mailbox_ts
            self.next_mail_pos = torch.zeros((num_nodes), dtype=torch.long) if _next_mail_pos is None else _next_mail_pos
        self.update_mail_pos = _update_mail_pos
        self.device = torch.device('cpu')

        self.feat_buffer = feat_buffer
        
    def set_buffer(self, feat_buffer):
        self.feat_buffer = feat_buffer

        if (self.prefetch_conn is None):
            self.feat_buffer.init_memory(self.node_memory, self.node_memory_ts, self.mailbox, self.mailbox_ts)

    def reset(self):
        if (self.prefetch_conn):
            self.prefetch_conn.send(('reset_memory', ()))
            self.prefetch_conn.recv()
        else:
            self.node_memory.fill_(0)
            self.node_memory_ts.fill_(0)
            self.mailbox.fill_(0)
            self.mailbox_ts.fill_(0)
            self.next_mail_pos.fill_(0)

    def move_to_gpu(self):
        self.node_memory = self.node_memory.cuda()
        self.node_memory_ts = self.node_memory_ts.cuda()
        self.mailbox = self.mailbox.cuda()
        self.mailbox_ts = self.mailbox_ts.cuda()
        self.next_mail_pos = self.next_mail_pos.cuda()
        self.device = torch.device('cuda:0')

    def allocate_pinned_memory_buffers(self, sample_param, batch_size):
        limit = int(batch_size * 3.3)
        if 'neighbor' in sample_param:
            for i in sample_param['neighbor']:
                limit *= i + 1
        self.pinned_node_memory_buffs = list()
        self.pinned_node_memory_ts_buffs = list()
        self.pinned_mailbox_buffs = list()
        self.pinned_mailbox_ts_buffs = list()
        for _ in range(sample_param['history']):
            self.pinned_node_memory_buffs.append(torch.zeros((limit, self.node_memory.shape[1]), pin_memory=True))
            self.pinned_node_memory_ts_buffs.append(torch.zeros((limit,), pin_memory=True))
            self.pinned_mailbox_buffs.append(torch.zeros((limit, self.mailbox.shape[1], self.mailbox.shape[2]), pin_memory=True))
            self.pinned_mailbox_ts_buffs.append(torch.zeros((limit, self.mailbox_ts.shape[1]), pin_memory=True))

    def prep_input_mails(self, mfg, use_pinned_buffers=False):
        for i, b in enumerate(mfg):
            if use_pinned_buffers:
                idx = b.srcdata['ID'].cpu().long()
                torch.index_select(self.node_memory, 0, idx, out=self.pinned_node_memory_buffs[i][:idx.shape[0]])
                b.srcdata['mem'] = self.pinned_node_memory_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                torch.index_select(self.node_memory_ts,0, idx, out=self.pinned_node_memory_ts_buffs[i][:idx.shape[0]])
                b.srcdata['mem_ts'] = self.pinned_node_memory_ts_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
                torch.index_select(self.mailbox, 0, idx, out=self.pinned_mailbox_buffs[i][:idx.shape[0]])
                b.srcdata['mem_input'] = self.pinned_mailbox_buffs[i][:idx.shape[0]].reshape(b.srcdata['ID'].shape[0], -1).cuda(non_blocking=True)
                torch.index_select(self.mailbox_ts, 0, idx, out=self.pinned_mailbox_ts_buffs[i][:idx.shape[0]])
                b.srcdata['mail_ts'] = self.pinned_mailbox_ts_buffs[i][:idx.shape[0]].cuda(non_blocking=True)
            else:
                if (self.feat_buffer != None and self.feat_buffer.mode == 'train'):
                    self.feat_buffer.input_mails(b)
                else:
                    if (self.feat_buffer != None and self.feat_buffer.prefetch_conn != None):
                        b.srcdata['mem'] = self.feat_buffer.select_index('memory',b.srcdata['ID'].long()).cuda()
                        b.srcdata['mem_ts'] = self.feat_buffer.select_index('memory_ts', b.srcdata['ID'].long()).cuda()
                        b.srcdata['mem_input'] = self.feat_buffer.select_index('mailbox', b.srcdata['ID'].long()).cuda().reshape(b.srcdata['ID'].shape[0], -1)
                        b.srcdata['mail_ts'] = self.feat_buffer.select_index('mailbox_ts', b.srcdata['ID'].long()).cuda()
                    else:
                        b.srcdata['mem'] = self.node_memory[b.srcdata['ID'].long()].cuda()
                        b.srcdata['mem_ts'] = self.node_memory_ts[b.srcdata['ID'].long()].cuda()
                        b.srcdata['mem_input'] = self.mailbox[b.srcdata['ID'].long()].cuda().reshape(b.srcdata['ID'].shape[0], -1)
                        b.srcdata['mail_ts'] = self.mailbox_ts[b.srcdata['ID'].long()].cuda()

    def update_memory(self, nid, memory, root_nodes, ts, neg_samples=1):
        if nid is None:
            return

        if (self.feat_buffer != None and self.feat_buffer.mode == 'train'):
            device = 'cuda:0'
        else:
            device = self.device

        num_true_src_dst = root_nodes.shape[0] // (neg_samples + 2) * 2
        with torch.no_grad():
            nid = nid[:num_true_src_dst].to(device)
            memory = memory[:num_true_src_dst].to(device)
            ts = ts[:num_true_src_dst].to(device)
            #TODO 优化? 此处的nid并不是unique的，也就是说并没有取最新的memory？ 我们手动取一个unique
            uni, inv = torch.unique(nid, return_inverse=True)
            perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
            perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
            nid = uni
            memory = memory[perm]
            ts = ts[perm]

            if (self.feat_buffer != None and self.feat_buffer.mode == 'train'):
               
                nid = torch.unique(nid)
                self.feat_buffer.update_memory(nid, memory, ts)

                # self.node_memory[nid.long().cpu()] = memory.cpu()
                # self.node_memory_ts[nid.long().cpu()] = ts.cpu()
            else:
                if (self.feat_buffer != None and self.feat_buffer.prefetch_conn != None):
                    self.feat_buffer.update_index('memory', nid.long(), memory.cpu())
                    self.feat_buffer.update_index('memory_ts', nid.long(), ts.cpu())
                else:
                    self.node_memory[nid.long()] = memory
                    self.node_memory_ts[nid.long()] = ts

    def update_mailbox(self, nid, memory, root_nodes, ts, edge_feats, block, neg_samples=1):
        if (self.feat_buffer != None and self.feat_buffer.mode == 'train'):
            device = 'cuda:0'
        else:
            device = self.device
        with torch.no_grad():
            num_true_edges = root_nodes.shape[0] // (neg_samples + 2)
            memory = memory.to(device)
            if edge_feats is not None:
                edge_feats = edge_feats.to(device)
            if block is not None:
                block = block.to(self.device)
            # TGN/JODIE
            if self.memory_param['deliver_to'] == 'self':
                src = torch.from_numpy(root_nodes[:num_true_edges]).to(device)
                dst = torch.from_numpy(root_nodes[num_true_edges:num_true_edges * 2]).to(device)
                mem_src = memory[:num_true_edges]
                mem_dst = memory[num_true_edges:num_true_edges * 2]
                #src和dst是源节点正边

                if self.dim_edge_feat > 0:
                    src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
                else:
                    src_mail = torch.cat([mem_src, mem_dst], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src], dim=1)
                    
                mail = torch.cat([src_mail, dst_mail], dim=1).reshape(-1, src_mail.shape[1])
                nid = torch.cat([src.unsqueeze(1), dst.unsqueeze(1)], dim=1).reshape(-1)
                mail_ts = torch.from_numpy(ts[:num_true_edges * 2]).to(device)
                if mail_ts.dtype == torch.float64:
                    import pdb; pdb.set_trace()
                # find unique nid to update mailbox
                uni, inv = torch.unique(nid, return_inverse=True)
                perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
                perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
                #perm的意义是找到nid中unique且在最后一个出现的，目的就是用所有节点的最新的那个做更新
                nid = nid[perm]
                mail = mail[perm]
                mail_ts = mail_ts[perm]

                if self.memory_param['mail_combine'] == 'last':
                    if (self.feat_buffer != None and self.feat_buffer.mode == 'train'):
                        self.feat_buffer.update_mailbox(nid, mail, mail_ts)

                        # self.mailbox[nid.long().cpu(), self.next_mail_pos[nid.long()].cpu()] = mail.cpu()
                        # self.mailbox_ts[nid.long().cpu(), self.next_mail_pos[nid.long()].cpu()] = mail_ts.cpu()
                        # if self.memory_param['mailbox_size'] > 1:
                        #     self.next_mail_pos[nid.long()] = torch.remainder(self.next_mail_pos[nid.long()] + 1, self.memory_param['mailbox_size'])

                    else:
                        if (self.feat_buffer != None and self.feat_buffer.prefetch_conn != None):
                            self.feat_buffer.update_index('mailbox', (nid.long(), 0), mail.cpu())
                            self.feat_buffer.update_index('mailbox_ts', (nid.long(), 0), mail_ts.cpu())
                        else:
                            self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail
                            self.mailbox_ts[nid.long(), self.next_mail_pos[nid.long()]] = mail_ts

                        #TODO 处理大于1的情况,但不是很想处理
                        if self.memory_param['mailbox_size'] > 1:
                            self.next_mail_pos[nid.long()] = torch.remainder(self.next_mail_pos[nid.long()] + 1, self.memory_param['mailbox_size'])
            # APAN
            elif self.memory_param['deliver_to'] == 'neighbors':
                mem_src = memory[:num_true_edges]
                mem_dst = memory[num_true_edges:num_true_edges * 2]
                if self.dim_edge_feat > 0:
                    src_mail = torch.cat([mem_src, mem_dst, edge_feats], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src, edge_feats], dim=1)
                else:
                    src_mail = torch.cat([mem_src, mem_dst], dim=1)
                    dst_mail = torch.cat([mem_dst, mem_src], dim=1)
                mail = torch.cat([src_mail, dst_mail], dim=0)
                mail = torch.cat([mail, mail[block.edges()[0].long()]], dim=0)
                mail_ts = torch.from_numpy(ts[:num_true_edges * 2]).to(self.device)
                mail_ts = torch.cat([mail_ts, mail_ts[block.edges()[0].long()]], dim=0)
                if self.memory_param['mail_combine'] == 'mean':
                    (nid, idx) = torch.unique(block.dstdata['ID'], return_inverse=True)
                    mail = scatter(mail, idx, reduce='mean', dim=0)
                    mail_ts = scatter(mail_ts, idx, reduce='mean')
                    self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail
                    self.mailbox_ts[nid.long(), self.next_mail_pos[nid.long()]] = mail_ts
                elif self.memory_param['mail_combine'] == 'last':
                    nid = block.dstdata['ID']
                    # find unique nid to update mailbox
                    uni, inv = torch.unique(nid, return_inverse=True)
                    perm = torch.arange(inv.size(0), dtype=inv.dtype, device=inv.device)
                    perm = inv.new_empty(uni.size(0)).scatter_(0, inv, perm)
                    nid = nid[perm]
                    mail = mail[perm]
                    mail_ts = mail_ts[perm]
                    self.mailbox[nid.long(), self.next_mail_pos[nid.long()]] = mail
                    self.mailbox_ts[nid.long(), self.next_mail_pos[nid.long()]] = mail_ts
                else:
                    raise NotImplementedError
                
                if self.memory_param['mailbox_size'] > 1:
                    if self.update_mail_pos is None:
                        self.next_mail_pos[nid.long()] = torch.remainder(self.next_mail_pos[nid.long()] + 1, self.memory_param['mailbox_size'])
                    else:
                        self.update_mail_pos[nid.long()] = 1
            else:
                raise NotImplementedError

    def update_next_mail_pos(self):
        if self.update_mail_pos is not None:
            nid = torch.where(self.update_mail_pos == 1)[0]
            self.next_mail_pos[nid] = torch.remainder(self.next_mail_pos[nid] + 1, self.memory_param['mailbox_size'])
            self.update_mail_pos.fill_(0)

class FFNMemeoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat):
        super(FFNMemeoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_in = dim_in
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.updater = torch.nn.Linear(dim_in + dim_time, dim_hid)
        self.layer_norm = torch.nn.LayerNorm(dim_hid)
        self.layer_norm1 = torch.nn.LayerNorm(dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)
        

    def forward(self, mfg):
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
           
            updated_memory = self.layer_norm(self.updater(b.srcdata['mem_input'])) + b.srcdata['mem']
            updated_memory = self.layer_norm1(updated_memory)
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            if self.memory_param['combine_node_feature']:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += updated_memory
                    else:
                        b.srcdata['h'] = updated_memory + self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = updated_memory

class GRUMemeoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat):
        super(GRUMemeoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.updater = torch.nn.GRUCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)

    #forward的是最下面那层的k-hop邻域节点
    def forward(self, mfg):
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
            updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
            self.last_updated_ts = b.srcdata['ts'].detach()
            self.last_updated_memory = updated_memory.detach()
            self.last_updated_nid = b.srcdata['ID'].detach()
            if self.memory_param['combine_node_feature']:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += updated_memory
                    else:
                        b.srcdata['h'] = updated_memory + self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = updated_memory

class RNNMemeoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_hid, dim_time, dim_node_feat):
        super(RNNMemeoryUpdater, self).__init__()
        self.dim_hid = dim_hid
        self.dim_node_feat = dim_node_feat
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.updater = torch.nn.RNNCell(dim_in + dim_time, dim_hid)
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        if memory_param['combine_node_feature']:
            if dim_node_feat > 0 and dim_node_feat != dim_hid:
                self.node_feat_map = torch.nn.Linear(dim_node_feat, dim_hid)

    def forward(self, mfg):
        for b in mfg:
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'] - b.srcdata['mem_ts'])
                b.srcdata['mem_input'] = torch.cat([b.srcdata['mem_input'], time_feat], dim=1)
            updated_memory = self.updater(b.srcdata['mem_input'], b.srcdata['mem'])
            self.last_updated_ts = b.srcdata['ts'].detach().clone()
            self.last_updated_memory = updated_memory.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            if self.memory_param['combine_node_feature']:
                if self.dim_node_feat > 0:
                    if self.dim_node_feat == self.dim_hid:
                        b.srcdata['h'] += updated_memory
                    else:
                        b.srcdata['h'] = updated_memory + self.node_feat_map(b.srcdata['h'])
                else:
                    b.srcdata['h'] = updated_memory

class TransformerMemoryUpdater(torch.nn.Module):

    def __init__(self, memory_param, dim_in, dim_out, dim_time, train_param):
        super(TransformerMemoryUpdater, self).__init__()
        self.memory_param = memory_param
        self.dim_time = dim_time
        self.att_h = memory_param['attention_head']
        if dim_time > 0:
            self.time_enc = TimeEncode(dim_time)
        self.w_q = torch.nn.Linear(dim_out, dim_out)
        self.w_k = torch.nn.Linear(dim_in + dim_time, dim_out)
        self.w_v = torch.nn.Linear(dim_in + dim_time, dim_out)
        self.att_act = torch.nn.LeakyReLU(0.2)
        self.layer_norm = torch.nn.LayerNorm(dim_out)
        self.mlp = torch.nn.Linear(dim_out, dim_out)
        self.dropout = torch.nn.Dropout(train_param['dropout'])
        self.att_dropout = torch.nn.Dropout(train_param['att_dropout'])
        self.last_updated_memory = None
        self.last_updated_ts = None
        self.last_updated_nid = None

    def forward(self, mfg):
        for b in mfg:
            Q = self.w_q(b.srcdata['mem']).reshape((b.num_src_nodes(), self.att_h, -1))
            mails = b.srcdata['mem_input'].reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], -1))
            if self.dim_time > 0:
                time_feat = self.time_enc(b.srcdata['ts'][:, None] - b.srcdata['mail_ts']).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], -1))
                mails = torch.cat([mails, time_feat], dim=2)
                
            K = self.w_k(mails).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1))
            V = self.w_v(mails).reshape((b.num_src_nodes(), self.memory_param['mailbox_size'], self.att_h, -1))


            att = self.att_act((Q[:,None,:,:]*K).sum(dim=3))
            att = torch.nn.functional.softmax(att, dim=1)
            att = self.att_dropout(att)
            rst = (att[:,:,:,None]*V).sum(dim=1)
            rst = rst.reshape((rst.shape[0], -1))
            rst += b.srcdata['mem']
            rst = self.layer_norm(rst)
            rst = self.mlp(rst)
            rst = self.dropout(rst)
            rst = torch.nn.functional.relu(rst)
            b.srcdata['h'] = rst
            self.last_updated_memory = rst.detach().clone()
            self.last_updated_nid = b.srcdata['ID'].detach().clone()
            self.last_updated_ts = b.srcdata['ts'].detach().clone()

