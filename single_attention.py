import numpy as np
import math
L,d_k,d_v=4,8,8
q=np.random.randn(L,d_k)
k=np.random.randn(L,d_k)
v=np.random.randn(L,d_v)
print("Q\n",q)
print("K\n",k)
print("V\n",v)

def softmax(x):
    return (np.exp(x).T/np.sum(np.exp(x),axis=-1)).T
mask =np.tril(np.ones((L,L)))
mask[mask==0] = -np.inf
mask[mask==1]=0

def scaled_dot_product_attention(q,k,v, mask=None):
    d_k=q.shape[-1]
    scaled=np.matmul(q,k.T)/math.sqrt(d_k)
    if mask is not None:
        scaled += mask
    attention=softmax(scaled)
    out=np.matmul(attention,v)
    return out,attention
values ,attention=scaled_dot_product_attention(q,k,v ,mask=None)
print("Q\n",q)
print("K\n",k)
print("V\n",v)
print("new v\n",values)
print("attention\n",attention)