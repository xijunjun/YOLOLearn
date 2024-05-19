import torch

# 源维度 1,3,6400,21   目标维度 1,63,80,80
# 为什么要这样设置维度？
# 目的：希望out[0,0,0,:]出来的就是单个通道。希望out[0,1,0,:]就是指定第1个a
# 【记住这个要领基本都能搞定】permute是改变访问顺序，reshape是调整形状不改变顺序。

if __name__=='__main__':
    
    tensor_zeros = torch.rand(1, 63, 80, 80)
    tensor_zeros_new=tensor_zeros.view(1,3,21,-1).permute(0,1,3,2)

    tensor_zeros =tensor_zeros.view(-1,21,80,80)  #(3,21,80,80)
    tensor_zeros=tensor_zeros.permute(0,2,3,1)#(3,80,80,21)
    tensor_zeros =tensor_zeros.view(1,3,-1,21)#(1,3,6400,21)

    print(tensor_zeros.shape)
    print(tensor_zeros_new.shape)

    print('finish')