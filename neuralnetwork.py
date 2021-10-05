# 开发者 haotian
# 开发时间: 2021/9/22 14:24
import math,random

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

#矩阵的加法
def Matrix_addition(a,b):
    t = [[0] * len(a[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            t[i][j] = a[i][j] + b[i][j]
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

#一个数乘一个矩阵
def Matrixa_num(a:float,b:list):
    t = [[0]*len(b[0]) for _ in range(len(b))]
    for i in range(len(b)):
        for j in range(len(b[0])):
            t[i][j] = a * b[i][j]
    return t

#每层权重自己定义
def init_nextwork():
    network = list()
    network.append([0])
    #随机初始化
    # network.append([[0.1,0.5,0.4],[0.2,0.3,0.5],[0.3,0.4,0.3]])
    # network.append([[0.3,0.5,0.2],[0.1,0.1,0.8],[0.4,0.4,0.2]])
    # network.append([[0.2,0.6,0.2]])
    network.append([[random.random(), random.random(), random.random()], [random.random(), random.random(), random.random()], [random.random(), random.random(), random.random()]])
    network.append([[random.random(), random.random(), random.random()], [random.random(), random.random(), random.random()], [random.random(), random.random(), random.random()]])
    network.append([[random.random(), random.random(), random.random()]])
    return network

#前向传播
def forward(network,a,f):
    #a代表每一层的激活值
    w1,w2,w3 = network[1],network[2],network[3]
    #第二层激活值
    if f == 0:
        a.append(sigmoid(w1,a[1]))
        # a[2] = sigmoid(w1,x)
        a.append(sigmoid(w2,a[2]))
        # a[3] = sigmoid(w2,a[2])
        a.append(sigmoid(w3,a[3]))
        # a[4] = sigmoid(w3,a[3])
    else:
        a[2] = sigmoid(w1,a[1])
        a[3] = sigmoid(w2,a[2])
        a[4] = sigmoid(w3,a[3])

    return a[4]

def backward(d,f):
    #实际上不能直接d[4]=xxxx 要用append 实际应用的时候再改。
    #这里y是样例的准确输出值
    #d代表每层的误差
    #假定的只有一个 y所以直接用减法就可以了
    #d[4]
    if f == 0:
        d.append([[a[4][0]-y[0][0]]])
        # d[4] = y - a[4]
        #这里 d = w的转置*d(前一个） 点乘 激活函数对d的求导，
        #后面的求导又 等于 a 点乘 （1-a)
        #d[3]
        d.append(Matrixa_dot(Matrixa_multiplication(transpose(network[3]),d[0]),
                             Matrixa_dot(transpose([a[3]]),sub([[1],[1],[1]],transpose([a[3]])))))
        #d[2]
        d.append(Matrixa_dot(Matrixa_multiplication(transpose(network[2]), d[1]),
                             Matrixa_dot(transpose([a[2]]), sub([[1], [1], [1]], transpose([a[2]])))))
        d.append([[0],[0],[0]])
        d.append([[0]])
        d.reverse()
    else:
        d[4] = [[a[4][0]-y[f%m][0]]]
        d[3] = Matrixa_dot(Matrixa_multiplication(transpose(network[3]),d[4]),
                             Matrixa_dot(transpose([a[3]]),sub([[1],[1],[1]],transpose([a[3]]))))
        d[2] = Matrixa_dot(Matrixa_multiplication(transpose(network[2]),d[3]),
                             Matrixa_dot(transpose([a[2]]), sub([[1], [1], [1]], transpose([a[2]]))))
        pass

    #计算D[i][j]
    #为什么会有警告？？
    # print(len(network[3][0]))

    #以下两种写法得到的D相同
    # for i in range(n):
    #     for j in range(len(network[n-i])):
    #         for k in range(len(network[n-i][0])):
    #             D[n-i-1][j][k] += a[n-i][k]*d[n-i+1][j][0]*(1/m)
    # print(D)

    #这样得到的D不适合单个元素的改变，一开始没设置矩阵的大小所以不能用 [i][j]来直接定位

    if f == 0:
        for i in range(n):
            D.append(Matrixa_multiplication(d[n-i+1],[a[n-i]]))
        D.reverse()
    else:
        for i in range(n):
            D[n-i-1] = Matrix_addition(D[n-i-1],Matrixa_multiplication(d[n-i+1],[a[n-i]]))

    # print(D)



#
def partial_derivative(f):
    if f == 0:
        for i in range(n):
            DD.append(Matrix_addition(Matrixa_num(1/m,D[i]),Matrixa_num(lan,network[i+1])))
    # print(DD)


if __name__ == '__main__':


    #一开始不初始化list，第一次使用要用append，以后就不用了

    #正则化参数
    lan :float = 0.1
    #n神经网络层数
    n :int = 3
    # f标志是否是第一次循环
    # 激活值
    a = list()
    # 每层误差
    d = list()
    # D 偏导数？？
    # D = [[[0]*3 for _ in range(3) ]for _ in range(3)]
    D = list()
    #r 学习率
    r : float = 0.7
    #偏导数
    DD = list()
    #初始化权重
    network = init_nextwork()
    #输入值,应该是列向量，方便起见，设为行向量，注意转换。
    x = [[0.1,0.1,0.1],[0.9,0.9,0.9],[0.2,0.2,0.2],[0.3,0.3,0.3],[0.4,0.4,0.4]]
    y = [[0.1],[0.9],[0.2],[0.3],[0.4]]
    # m数据个数
    m = len(x)
    flage :int = 0
    for j in range(100):
        for i in range(m):
            if flage == 0:
                a.append([0])
                #激活值从a[1]一开始，第一层是输入的值
                a.append(x[i])
            else:
                a[1] = x[i]
            forward(network, a, flage)
            backward(d, j)
            flage = 1
        partial_derivative(j)
        for i in range(3):
            network[i+1] = Matrix_addition(network[i+1],Matrixa_num(-r,DD[i]))
        # print(network)
        a[1] = [0.6,0.6,0.6]
    print(forward(network,a,11))

