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

#矩阵的转置
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

#矩阵的减法,对应元素相减
def sub(a,b):
    t = [[0] * len(a[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            t[i][j] = a[i][j] - b[i][j]
    return t

#矩阵的乘法
def Matrixa_multiplication(a,b:list):
    #这里当b[0]只有一个数时，len方法是用了的
    # print(b)
    t = [[0]*len(b[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            tt :int = 0
            for k in range(len(a[0])):
                tt += a[i][k]*b[k][j]
            t[i][j] = tt
    return t

#矩阵的点乘，对应元素相乘
def Matrixa_dot(a,b:list):
    t = [[0]*len(a[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            t[i][j] = a[i][j]*b[i][j]
    return t

#每层权重自己定义
def init_nextwork():
    network = {}
    #两个隐藏层各有三个节点 ，权重自己设置的
    network['w1'] = [[0.1,0.5,0.4],[0.2,0.3,0.5],[0.3,0.4,0.3]]
    network['w2'] = [[0.3,0.5,0.2],[0.1,0.1,0.8],[0.4,0.4,0.2]]
    #输出一个y
    network['w3'] = [[0.2,0.6,0.2]]
    return network
def forward(network,a):
    #a代表每一层的激活值
    w1,w2,w3 = network['w1'],network['w2'],network['w3']
    #第二层激活值
    a.append(sigmoid(w1,a[1]))
    # a[2] = sigmoid(w1,x)
    a.append(sigmoid(w2,a[2]))
    # a[3] = sigmoid(w2,a[2])
    a.append(sigmoid(w3,a[3]))
    # a[4] = sigmoid(w3,a[3])
    return a[4]

def backward(d):
    #实际上不能直接d[4]=xxxx 要用append 实际应用的时候再改。
    #这里y是样例的准确输出值
    #d代表每层的误差
    #假定的只有一个 y所以直接用减法就可以了
    #d[4]
    d.append([[a[4][0]-y[0]]])
    # d[4] = y - a[4]
    #这里 d = w的转置*d(前一个） 点乘 激活函数对d的求导，
    #后面的求导又 等于 a 点乘 （1-a)
    #d[3]
    d.append(Matrixa_dot(Matrixa_multiplication(transpose(network['w3']),d[0]),
                         Matrixa_dot(transpose([a[3]]),sub([[1],[1],[1]],transpose([a[3]])))))
    #d[2]
    d.append(Matrixa_dot(Matrixa_multiplication(transpose(network['w2']), d[1]),
                         Matrixa_dot(transpose([a[2]]), sub([[1], [1], [1]], transpose([a[2]])))))
    d.append([[0],[0],[0]])
    d.append([[0]])
    d.reverse()





if __name__ == '__main__':
    # 激活值
    a = list()
    # 每层误差
    d = list()
    # D
    D = list()
    #输入值,应该是列向量，方便起见，设为行向量，注意转换。
    x = [0.1,0.2,0.3]
    y = [0.6]
    a.append([0])
    #激活值从a[1]一开始，第一层是输入的值
    a.append(x)
    # a[1] = x
    network = init_nextwork()
    forward(network,a)
    backward(d)
    #print(sub([[1,2],[3,4]],[[1,2],[3,4]]))
    # l = [[1,2,3],[4,5,6]]
    # print(transpose(l))
    # print(Matrixa_multiplication(l,transpose(l)))
    # print(Matrixa_dot(l,l))
