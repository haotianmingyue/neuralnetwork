# 开发者 haotian
# 开发时间: 2021/9/22 14:24
import math

'''
sigmoid激活函数
    w代表权重，w的第一行代表得到第一个输出的权重
    x代表输入的值
    返回一个数组
'''
def sigmoid(w,x):
    t = list()
    for i in range(len(w)):
        n_x :int = 0
        for j in range(len(w[i])):
            n_x +=w[i][j]*x[j]
        t.append(1/(1+math.exp(-n_x)))
    return t

def transpose(t):
    #原先矩阵行数
    m = len(t)
    #原先矩阵列数
    n = len(t[0])
    n_t =[[0]*m for _ in range(n)]
    for i in range(m):
        for j in range(n):
            n_t[j][i] = t[i][j]
    return n_t

#矩阵的减法
def sub(a,b):
    t = [[0] * len(b[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            tt: int = 0
            for k in range(len(a[0])):
                tt += a[i][k] - b[k][j]
            t[i][j] = tt
    return t

#矩阵的乘法
def Matrixa_multiplication(a,b):
    t = [[0]*len(b[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            tt :int = 0
            for k in range(len(a[0])):
                tt += a[i][k]*b[k][j]
            t[i][j] = tt
    return t

#矩阵的点乘
def Matrixa_dot(a,b):
    t = [[0]*len(a[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            t[i][j] = a[i][j]*b[i][j]
    return t

#每层权重自己定义
def init_nextwork():
    network = {}
    #两个隐藏层各有三个节点
    network['w1'] = [[0.1,0.5,0.4],[0.2,0.3,0.5],[0.3,0.4,0.3]]
    network['w2'] = [[0.3,0.5,0.2],[0.1,0.1,0.8],[0.4,0.4,0.2]]
    #输出一个y
    network['w3'] = [[0.2,0.6,0.2]]
    return network
def forward(network,a):
    w1,w2,w3 = network['w1'],network['w2'],network['w3']
    a.append(sigmoid(w1,a[1]))
    # a[2] = sigmoid(w1,x)
    a.append(sigmoid(w2,a[2]))
    # a[3] = sigmoid(w2,a[2])
    a.append(sigmoid(w3,a[3]))
    # a[4] = sigmoid(w3,a[3])
    return a[4]

def backward(d):
    #这里y是样例的准确输出值
    #d代表每层的误差
    d[4] = y - a[4]
    d[3] = Matrixa_dot(Matrixa_multiplication(transpose(network['w3']),d[4]),Matrixa_dot(a[3],sub([1,1,1],a[3])))
    d[2] = Matrixa_dot(Matrixa_multiplication(transpose(network['w2']),d[3]),Matrixa_dot(a[2],sub([1,1,1],a[2])))
    pass

if __name__ == '__main__':
    a = list()
    #输入值
    x = [0.1,0.2,0.3]
    y = [0.01,0.04,0.09]
    a.append(0)
    a.append(x)
    # a[1] = x
    network = init_nextwork()
    print(forward(network,a))
    # l = [[1,2,3],[4,5,6]]
    # print(transpose(l))
    # print(Matrixa_multiplication(l,transpose(l)))
    # print(Matrixa_dot(l,l))
