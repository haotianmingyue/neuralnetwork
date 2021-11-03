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
from read_iris import read_iris_data


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
        # print('b[0]',b[0])
        t = [[0] * len(b[0]) for _ in range(len(a))]

        for i in range(len(a)):
            for j in range(len(b[0])):
                tt: int = 0
                for k in range(len(a[0])):
                    tt += a[i][k] * b[k][j]
                t[i][j] = tt
        return t

    # 矩阵的点乘，对应元素相乘
    def Matrixa_element_wise(self,a, b: list):
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
    #数值为n的列矩阵
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
    ):   #初始化网络  类成员有 net 每层的值， w整个的权重，delta 每层的误差，der_t 每层的导数 der每组数据的导数和
        self.net = list() # 整个网络架构,每层节点的值
        for i in range(len(l)):
            self.net.append([[0] for _ in range(l[i])])

        self.w = list() #权重矩阵 一共有 len(l)-1个变量
        for i in range(1,len(l)):
            self.w.append([[0] * l[i-1] for _ in range(l[i])])

        self.delta = copy.deepcopy(self.net)
        self.der = copy.deepcopy(self.w) #每组导数和
        self.der_t = copy.deepcopy(self.w) #一组的导数

        for i in range(len(self.w)):
            for j in range(l[i+1]):
                for k in range(l[i]):
                    self.w[i][j][k] = random.random()
        self.lr = learningrate


    def fit(self,x,y):
        layer = layers()
        m = len(x)
        for i in range(m):
            self.forward(self.w,x[i])
            self.backward(y[i])
            # print(self.loss(self.net,y[i]))
        layer.gradient_descent(m,self.w,self.der,self.der_t,0.7,self.lr)

        # print(self.w)

    #预测函数
    def predict(self,x):
        t = list()
        for i in range(len(x)):
            self.forward(self.w,x[i])
            print(self.net[-1])
            # t.append(self.net[-1])
        return t

    #激活函数
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
        # print('x',x)
        self.net[0] = x
        m = matrix()
        for i in range(1,len(self.net)):
            self.net[i] = self.sigmoid(m.Matrixa_mul(w[i-1], self.net[i-1]))

    def backward(self,y):
        layer = layers()
        layer.count_dlter(self.w,self.delta,self.net,y)
        layer.partial_derivative(self.der_t,self.delta,self.net)

    # 损失函数
    def loss(self,net,y):
        ls: float = 0
        for i in range(len(y)):
            ls += (net[-1][i][0] - y[i][0]) ** 2
        return ls


class layers():
    m = matrix()

    def count_dlter(self,w,delta,net,y):
        #最后一个误差直接减去y
        delta[-1] = self.m.Matrix_sub(net[-1], y)
        for i in range(len(delta)-2,0,-1):
            delta[i] = self.m.Matrixa_element_wise(self.m.Matrixa_element_wise(self.m.Matrixa_mul(self.m.transpose(w[i]), delta[i+1]), net[i]), self.m.Matrix_sub(self.m.one_matrixa(net[i], 1), net[i]))

    def partial_derivative(self,der_t,delta,net):
        for i in range(len(der_t)):
            der_t[i] = self.m.Matrix_add(der_t[i], self.m.Matrixa_mul(delta[i+1], self.m.transpose(net[i])))

    def gradient_descent(self,m,w,der,der_t,la,lr):
        for i in range(len(der)):
            der[i] = self.m.Matrix_add(self.m.Matrixa_num(1 / m, der_t[i]), self.m.Matrixa_num(la, w[i]))
        for i in range(len(w)):
            w[i] = self.m.Matrix_sub(w[i], self.m.Matrixa_num(lr, der[i]))





if __name__ == '__main__':
    #初始化数组 第一个数代表输入节点个数最后一数代表输出节点个数，中间代表每个隐层的节点个数，最后一个代表学习率
    net = network([4,4,4,3],0.4)
    x,y = read_iris_data()
    for i in range(1000):
        # print(x,y[i])
        #其中x，y为整个训练集，不要放单个数据
        net.fit(x, y)
    # print(y[100],y[22],y[33],y[44],y[123],y[125])
    #predict接收的是 整个待预测数据集，不要放单个数据
    # net.predict([x[100],x[22],x[33],x[44],x[123],x[125]])
    print(net.predict(x))










