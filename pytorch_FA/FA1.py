import torch



def tiled_forward(self ,q ,k ,v ,Bc ,Br ):
    N,d = q.shape
    Tc = (N + Bc - 1) // Bc #列块数
    Tr = (N + Br - 1) // Br #行块数
    
    O = torch.zeros_like(q)
    l = torch.zeros(N)
    m = torch.full((N,),float('-inf'))
    #外循环:遍历K,V的块
    for j in range(Tc):
        # load k,v块到SRAM
        k_block = k[j*Bc:min((j+1)*Bc,N)]
        v_block = v[j*Bc:min((j+1)*Bc,N)]

        #内循环:遍历Q的块
        for i in range(Tr):

            #load q 块到SRAM
            start_idx = i * Br
            end_idx = min((i+1)*Br,N)
            q_block = q[start_idx:end_idx]
            

            #in SRAM
            #计算局部注意力分数
            S_ij = self.sm_scale * (q_block @ k.block.T)

            #更新最大值
            #将m的对应的Br部分从HBM中load到SRAM中
            m_block = m[start_idx:end_idx]
            m_new = torch.max(torch.max(S_ij,dim=-1)[0]

            P_ij = torch.exp(S_ij - m_new.unsqueeze(1))
            l_block = l[start_idx:end_idx]
            l_new = torch.exp(m_block - m_new) * l_block

            O_block = O[start_idx:end_idx]
            O[start_idx:end_idx]=(
                torch.exp(m_block - m_new).unsqueeze(1)*O_block + P_ij @ v_block
                )/l_new.unsqueeze(1)

            #更新中间变量
            m[start_idx:end_idx] = m_new
            l[start_idx:end_idx] = l_new
    return 0
