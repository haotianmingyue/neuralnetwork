#!/usr/bin/env python3
# -*-coding:utf-8 -*-
# @anthor haotian
# @date 2021/10/21
# @file neural_network_object.py
# 开发者 haotian
# 开发时间: 2021/10/5 15:08
import copy
import random
import math


class matrix():
    # 矩阵的转置
    def transpose(self,t):
        # 原先矩阵行数
        m = len(t)
        # 原先矩阵列数
        n = len(t[0])
        n_t = [[0] * m for _ in range(n)]
        for i in range(m):
            for j in range(n):
                n_t[j][i] = t[i][j]
        return n_t

    # 矩阵的减法,对应元素相减
    def Matrix_sub(self,a, b):
        t = [[0] * len(a[0]) for _ in range(len(a))]
        for i in range(len(a)):
            for j in range(len(a[0])):
                t[i][j] = a[i][j] - b[i][j]
        return t

    # 矩阵的加法
    def Matrix_add(self,a, b):
        t = [[0] * len(a[0]) for _ in range(len(a))]
        for i in range(len(a)):
            for j in range(len(a[0])):
                t[i][j] = a[i][j] + b[i][j]
        return t

    # 矩阵的乘法
    def Matrixa_mul(self,a, b: list):
        t = [[0] * len(b[0]) for _ in range(len(a))]

        for i in range(len(a)):
            for j in range(len(b[0])):
                tt: int = 0
                for k in range(len(a[0])):
                    tt += a[i][k] * b[k][j]
                t[i][j] = tt
        return t

    # 矩阵的点乘，对应元素相乘
    def Matrixa_dot(self,a, b: list):
        t = [[0] * len(a[0]) for _ in range(len(a))]
        for i in range(len(a)):
            for j in range(len(a[0])):
                t[i][j] = a[i][j] * b[i][j]
        return t

    # 一个数乘一个矩阵
    def Matrixa_num(self,a: float, b: list):
        t = [[0] * len(b[0]) for _ in range(len(b))]
        for i in range(len(b)):
            for j in range(len(b[0])):
                t[i][j] = a * b[i][j]
        return t

    def one_matrixa(self,a, n):
        t = list()
        for i in range(len(a)):
            tt = list()
            for j in range(len(a[0])):
                tt.append(n)
            t.append(tt)
        return t

class network():
    def __init__(
            self
            ,l  #输入行矩阵第一行代表输入节点个数，最后一行代表输出节点个数，中间代表隐层层数和每层个数
            , learningrate
    ):   #初始化网络  类成员有 net 每层的值， w整个的权重，delta 每层的误差，der 每层的导数
        self.net = list() # 整个网络架构,每层节点的值
        for i in range(len(l)):
            self.net.append([[0] for _ in range(l[i])])

        self.w = list() #权重矩阵 一共有 len(l)-1个变量
        for i in range(1,len(l)):
            self.w.append([[0] * l[i-1] for _ in range(l[i])])

        self.delta = copy.deepcopy(self.w)
        self.der = copy.deepcopy(self.w)

        for i in range(len(self.w)):
            for j in range(l[i+1]):
                for k in range(l[i]):
                    self.w[i][j][k] = random.random()
        self.lr = learningrate


    def fit(self,x):
        self.forward(self.w,x)

        pass

    # 迭代次数多了 y急速减小。。。会无限小。。
    def sigmoid(self,w):
        try:
            t = list()
            for i in range(len(w)):
                t.append([1 / (1 + math.exp(-w[i][0]))])
            return t
        except Exception as ex:
            print(ex)
            print(w)

    def forward(self,w, x):

        self.net[0] = x
        m = matrix()
        for i in range(1,len(self.net)):
            self.net[i] = self.sigmoid(m.Matrixa_mul(w[i-1], self.net[i-1]))



    def backward(self):


        pass



class layers():

    def count_dlter(self,d, j):
        d[3] = Matrix_sub(y, yy[j])
        d[2] = Matrixa_dot(Matrixa_dot(Matrixa_mul(transpose(w[2]), d[3]), c), Matrix_sub(one_matrixa(c, 1), c))
        d[1] = Matrixa_dot(Matrixa_dot(Matrixa_mul(transpose(w[1]), d[2]), c), Matrix_sub(one_matrixa(b, 1), b))

        pass

    def partial_derivative(self,D, j):
        D[2] = Matrix_add(D[2], Matrixa_mul(d[3], transpose(c)))
        D[1] = Matrix_add(D[1], Matrixa_mul(d[2], transpose(b)))
        D[0] = Matrix_add(D[0], Matrixa_mul(d[1], transpose(xx[j])))

        pass

    def gradient_descent(self,m, la):
        DD[2] = Matrix_add(Matrixa_num(1 / m, D[2]), Matrixa_num(la, w[2]))
        DD[1] = Matrix_add(Matrixa_num(1 / m, D[1]), Matrixa_num(la, w[1]))
        DD[0] = Matrix_add(Matrixa_num(1 / m, D[0]), Matrixa_num(la, w[0]))

        w[2] = Matrix_sub(w[2], Matrixa_num(lr, DD[2]))
        w[1] = Matrix_sub(w[1], Matrixa_num(lr, DD[1]))
        w[0] = Matrix_sub(w[0], Matrixa_num(lr, DD[0]))






#损失函数
def loss(y,j):
    ls :float = 0

    for i in range(len(y)):
        ls += (y[i][0] - yy[j][i][0])**2
    return ls


if __name__ == '__main__':
    # d,w,lr,delta,der =
    net = network([3,3,4,5],0.4)
    #d代表每层激活后的值，w代表权重矩阵，lr代表学习率，delta代表每层误差，der代表每层导数

    net.fit([[1],[1],[1]])

    # xx = list()  #输入数据
    # yy = list()  #输出数据





    # #xx yy 均为列向量
    # xx = list()
    # yy = list()
    #
    # for i in range(0,100):
    #
    #     t = list('{:07b}'.format(i))
    #     ttt = list()
    #     for tt in t:
    #         ttt.append([int(tt)])
    #     yy.append(ttt)
    #     xx.append(ttt)
    #
    #
    #
    # m = len(xx)
    #
    # for i in range(1000):
    #     ls: float = 0
    #     for j in range(m):
    #         b,c,y = forward(w,xx[j])
    #         backward(d,j)
    #         partial_derivative(D,j)
    #         ls += loss(y,j)
    #     print(ls/m)
    #     gradient_descent(m,0.7)
    # b,c,y =forward(w,xx[1])
    # print(y)
    # print(yy[1])







