def forward(self, q, k, v, key_padding_mask=None):
    """
    Flash Attention V2前向传播
    
    Args:
        q: 查询矩阵 [batch_size, seq_len_q, num_heads, head_dim]
        k: 键矩阵 [batch_size, seq_len_k, num_heads, head_dim]
        v: 值矩阵 [batch_size, seq_len_k, num_heads, head_dim]
        key_padding_mask: 键的填充掩码
    """
    batch_size, seq_len_q, num_heads, head_dim = q.shape
    seq_len_k = k.shape[1]
    
    # 初始化输出和中间结果
    output = torch.zeros_like(q)
    L = torch.zeros(batch_size, num_heads, seq_len_q, 
                   device=q.device, dtype=q.dtype)
    
    # 计算分块数量
    num_block_q = (seq_len_q + self.block_size_r - 1) // self.block_size_r
    num_block_k = (seq_len_k + self.block_size_c - 1) // self.block_size_c
    
    # 主要计算循环
    for i in range(num_block_q):
        # Q的当前块范围
        q_start = i * self.block_size_r
        q_end = min(q_start + self.block_size_r, seq_len_q)
        
        # 初始化当前Q块的状态
        q_block = q[:, q_start:q_end]
        m_i = torch.full((batch_size, num_heads, q_end-q_start), 
                        float('-inf'), device=q.device)
        l_i = torch.zeros((batch_size, num_heads, q_end-q_start), 
                         device=q.device)
        
        # 遍历K,V块
        for j in range(num_block_k):
            k_start = j * self.block_size_c
            k_end = min(k_start + self.block_size_c, seq_len_k)
            
            # 加载当前K,V块
            k_block = k[:, k_start:k_end]
            v_block = v[:, k_start:k_end]
            
            # 计算注意力分数
            S_ij = torch.matmul(q_block, k_block.transpose(-2, -1))
            if self.softmax_scale is not None:
                S_ij = S_ij * self.softmax_scale
                
            # 应用掩码（如果有）
            if key_padding_mask is not None:
                mask_block = key_padding_mask[:, k_start:k_end]
                S_ij = S_ij.masked_fill(~mask_block.unsqueeze(1).unsqueeze(2), 
                                      float('-inf'))
            
            # 更新最大值和累积和
            M_ij = torch.max(S_ij, dim=-1)[0]
            m_new = torch.max(m_i, M_ij)
            
            # 计算局部softmax
            exp_ij = torch.exp(S_ij - m_new.unsqueeze(-1))
            l_new = torch.exp(m_i - m_new) * l_i + torch.sum(exp_ij, dim=-1)
            
            # 更新输出
            output_block = output[:, q_start:q_end]
            output_block = torch.exp(m_i.unsqueeze(-1) - m_new.unsqueeze(-1)) * output_block
            output_block = output_block + torch.matmul(exp_ij, v_block)
            
            # 更新状态
            m_i = m_new
            l_i = l_new
            output[:, q_start:q_end] = output_block
            
        # 保存最终的logsumexp值
        L[:, :, q_start:q_end] = m_i + torch.log(l_i)
    
    # 最终归一化
    output = output / torch.exp(L).unsqueeze(-1)
    return output
