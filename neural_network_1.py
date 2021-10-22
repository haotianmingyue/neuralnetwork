# 开发者 haotian
# 开发时间: 2021/10/5 15:08
import copy
import random
import math

def init_network(inputnodes,hiddennode1,hiddennode2,outputnodes,learningrate):
    x = [[0] for _ in range(inputnodes)]
    b = [[0] for _ in range(hiddennode1)]
    c = [[0] for _ in range(hiddennode2)]
    y = [[0] for _ in range(outputnodes)]
    w = list()
    w.append([[0]*inputnodes for _ in range(hiddennode1)])
    w.append([[0]*hiddennode1 for _ in range(hiddennode2)])
    w.append([[0]*hiddennode2 for _ in range(outputnodes)])

    D = copy.deepcopy(w)
    DD = copy.deepcopy(w)

    for j in range(hiddennode1):
        for k in range(inputnodes):
            w[0][j][k] = random.random()
    for j in range(hiddennode2):
        for k in range(hiddennode1):
            w[1][j][k] = random.random()
    for j in range(outputnodes):
        for k in range(hiddennode2):
            w[2][j][k] = random.random()
    lr = learningrate
    return  [x,b,c,y],w,lr,D,DD

#迭代次数多了 y急速减小。。。会无限小。。
def sigmoid(w):
    try:
        t = list()
        for i in range(len(w)):
            t.append([1/(1+math.exp(-w[i][0]))])
        return t
    except Exception as ex:
        print(ex)
        print(w)

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
def Matrix_sub(a,b):
    t = [[0] * len(a[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            t[i][j] = a[i][j] - b[i][j]
    return t

#矩阵的加法
def Matrix_add(a,b):
    t = [[0] * len(a[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            t[i][j] = a[i][j] + b[i][j]
    return t

#矩阵的乘法
def Matrixa_mul(a,b:list):
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

def forward(w,x):
    #0.5 偏置
    # t = Matrixa_mul(w[0],x)
    # b = Matrix_add(sigmoid(t),one_matrixa(t,0.0001))
    # t = Matrixa_mul(w[1],b)
    # c = Matrix_add(sigmoid(t),one_matrixa(t,0.0001))
    # t = Matrixa_mul(w[2], c)
    # y = Matrix_add(sigmoid(t),one_matrixa(t,0.0001))



    b = sigmoid(Matrixa_mul(w[0],x))
    c = sigmoid(Matrixa_mul(w[1],b))
    y = sigmoid(Matrixa_mul(w[2],c))
    return b,c,y

def one_matrixa(a,n):
    t = list()
    for i in range(len(a)):
        tt = list()
        for j in range(len(a[0])):
            tt.append(n)
        t.append(tt)
    return t


def backward(d,j):
    d[3] = Matrix_sub(y,yy[j])
    d[2] = Matrixa_dot(Matrixa_dot(Matrixa_mul(transpose(w[2]),d[3]),c),Matrix_sub(one_matrixa(c,1),c))
    d[1] = Matrixa_dot(Matrixa_dot(Matrixa_mul(transpose(w[1]),d[2]),b),Matrix_sub(one_matrixa(b,1),b))

    pass

def partial_derivative(D,j):

    D[2] = Matrix_add(D[2],Matrixa_mul(d[3],transpose(c)))
    D[1] = Matrix_add(D[1],Matrixa_mul(d[2],transpose(b)))
    D[0] = Matrix_add(D[0],Matrixa_mul(d[1],transpose(xx[j])))

    pass

def gradient_descent(m,la):

    DD[2] = Matrix_add(Matrixa_num(1 / m, D[2]), Matrixa_num(la, w[2]))
    DD[1] = Matrix_add(Matrixa_num(1 / m, D[1]), Matrixa_num(la, w[1]))
    DD[0] = Matrix_add(Matrixa_num(1 / m, D[0]), Matrixa_num(la, w[0]))

    w[2] = Matrix_sub(w[2],Matrixa_num(lr,DD[2]))
    w[1] = Matrix_sub(w[1],Matrixa_num(lr,DD[1]))
    w[0] = Matrix_sub(w[0],Matrixa_num(lr,DD[0]))

#损失函数
def loss(y,j):
    ls :float = 0
    # print(y)
    # print(yy[j])
    for i in range(len(y)):
        ls += (y[i][0] - yy[j][i][0])**2
    return ls


if __name__ == '__main__':
    d,w,lr,D,DD = init_network(7,4,5,7,0.1)


    # xx = [[[0.4],[-4],[-0.4]],[[0.1],[0.1],[0.1]],[[0.9],[0.6],[0.5]]]
    # yy = [[[0],[1],[0]],[[1],[0],[0]],[[0],[0],[1]]]

    #xx yy 均为列向量
    xx = list()
    yy = list()

    for i in range(0,100):
        # xx.append([[i]])
        t = list('{:07b}'.format(i))
        ttt = list()
        for tt in t:
            ttt.append([int(tt)])
        yy.append(ttt)
        xx.append(ttt)

    # x1 = [[0.4],[-4],[-0.4]]
    # y1 = [[0],[1],[0]]

    m = len(xx)

    for i in range(1000):
        ls: float = 0
        for j in range(m):
            b,c,y = forward(w,xx[j])
            backward(d,j)
            partial_derivative(D,j)
            ls += loss(y,j)
        print(ls/m)
        gradient_descent(m,0.7)
    b,c,y =forward(w,xx[1])
    print(y)
    print(yy[1])

    # b,c,y = forward(w,xx[0])
    # print(y)
    # b,c,y = forward(w,xx[1])
    # print(y)
    # b,c,y = forward(w,xx[2])
    # print(y)





