# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
import copy

def sigmoid(X, diff=False):

    if diff:
        return sigmoid(X) * (1 - sigmoid(X))
    return 1 / (1 + np.exp(-X))

def gradient(y_parent, e):
    return y_parent * (1 - y_parent) * e




a = 0.1

input_size = 2
hidden_size = 2
output_size = 1
F = 2.4 / (input_size + hidden_size + output_size)

"""
weight_34 = np.random.rand(input_size * hidden_size).reshape(hidden_size, input_size)
bias_34 = np.random.rand(hidden_size).reshape(hidden_size, -1)

weight_34 = np.random.rand(hidden_size * output_size).reshape(output_size, hidden_size)
bias_34 = np.random.rand(output_size).reshape(output_size, -1)


weight_34 = np.array([[0.5, 0.4], [0.9, 1.0]])
weight_5 = np.array([[-1.2, 1.1]])

bias_34 = np.array([[0.8], [-0.1]])
bias_5 = np.array([[0.3]])
"""


"""
보고서 쓸 때,

1) 가중치와 임계값 수준을 (-2.4 / Fi , +2.4 / Fi)로 제한하자.
2) 그리고 균등분포를 사용
"""

class SingleLayer:
    def __init__(self, w, b, name):
        self.weight = w
        self.bias = b
        self.delta_weight = None
        self.delta_bias = None
        self.out = None # same in y
        self.name = name


class MLP:
    def __init__(self, input_size, hidden_size, output_size, name):
        # hidden size = ex) [2,3,5] . num of neruon, each hidden layer.
        self.a = 0.1
        self.name = name

        weight_input =  np.random.uniform(-F, F, input_size * hidden_size).reshape(hidden_size, input_size)
        bias_input =    np.random.uniform(-F, F, hidden_size).reshape(hidden_size, -1)

        weight_output = np.random.uniform(-F, F, hidden_size * output_size).reshape(output_size, hidden_size)
        bias_output = np.random.uniform(-F, F, output_size).reshape(output_size, -1)

        """
        self.weight_input = np.array([[0.5, 0.4], [0.9, 1.0]])
        self.bias_input = np.array([[0.8], [-0.1]])

        self.weight_output = np.array([[-1.2, 1.1]])
        self.bias_output = np.array([[0.3]])
        """


        self.layers = [ None, SingleLayer(weight_input, bias_input, "hidden1"), SingleLayer(weight_output, bias_output, "output") ]

    def test(self, train_data):

        self.layers.pop(0)
        self.layers.insert(0, train_data)

        prevLayer = None
        err = None
        X = None
        y = None
        for layer in self.layers:
            if type(layer) != SingleLayer:
                # prevLayer 중복지정 방지때문에 if then else로 처리
                pass
            else:
                X = self.step1_calDot(layer, prevLayer)
                y = self.step2_calSig(X)
                layer.out = y

            prevLayer = layer

        #err = self.step3_calErr(supervised, y)
        return y


    def errorRate(self, train_data ,supervised):
        y = self.test(train_data)

        return self.calcError(y, supervised)

    def calcError(self, y, supervised):

        return y - supervised

    def step1_calDot(self, layer, prevLayer):
        if type(prevLayer) != SingleLayer: # 이전층이 최초 인풋일경우
            result = np.dot(layer.weight, prevLayer) - layer.bias
            return result
        else:
            result = np.dot(layer.weight, prevLayer.out) - layer.bias
            return result




    def step2_calSig(self, X):
        return sigmoid(X)

    def step3_calErr(self, y, supervised):
        return supervised - y

    def step4_calGrad(self, y, prev_err):
        return gradient(y, prev_err)

    def step5_calDelta(self, learn_rate, postLayer, grad):
        if type(postLayer) != SingleLayer: # 바이어스도 해당
            result = learn_rate * postLayer * grad
            return result
        result = learn_rate * postLayer.out * grad
        return result


    def train(self, train_data, supervised):

        self.layers.pop(0)
        self.layers.insert(0, train_data)

        prevLayer = None
        err = None
        X = None
        y = None
        for layer in self.layers:
            if type(layer) != SingleLayer:
                # prevLayer 중복지정 방지때문에 if then else로 처리
                pass
            else:
                X = self.step1_calDot(layer, prevLayer)
                y = self.step2_calSig(X)
                layer.out = y


            prevLayer = layer

        err = self.step3_calErr(y, supervised)

        self.layers.reverse()
        prev_grad = err
        prev_layer = None
        #가중치 계산
        for idx, layer in enumerate(self.layers):
            if(idx == len(self.layers)-1): # input data까지 갔을 시.
                continue
            if(idx == 0): # output layer일 경우 바로 err
                prev_grad = self.step4_calGrad((layer.out).T, prev_grad)
            else:
                prev_grad = self.step4_calGrad((layer.out).T, prev_grad * prevLayer.weight)
            delta_W = self.step5_calDelta(self.a, self.layers[idx+1] ,prev_grad).T
            delta_b = self.step5_calDelta(self.a,  (-1), prev_grad).T
            layer.delta_weight = delta_W
            layer.delta_bias = delta_b

            prevlayer = layer

        # 가중치 실 적용
        for idx, layer in enumerate(self.layers):
            if (idx == len(self.layers) - 1):  # input data까지 갔을 시.
                continue

            layer.weight = layer.weight + layer.delta_weight
            layer.bias = layer.bias + layer.delta_bias

        # 원상복구
        self.layers.reverse()
        return prev_grad


    def train_desc(self, train_data, supervised):
        print(train_data, supervised)

        X34 = np.dot(self.weight_34, train_data) - self.bias_34
        print("X34 :")
        print(X34)

        y34 = sigmoid(X34)
        print("y34 :")
        print(y34)

        print("---------------------------")

        X5 = np.dot(self.weight_5, y34) - self.bias_5
        print("X5 :")
        print(X5)

        y5 = sigmoid(X5)
        print("y5 :")
        print(y5)
        print("---------------------------")

        final_err = supervised - y5
        print("final_err :")
        print(final_err)

        print("---------------------------")

        grad_5 = gradient(y5.T, final_err)
        print("grad_5 :")
        print(grad_5)

        print("---------------------------")

        delta_weight_5 = (a * y34 * grad_5).T
        print("delta_weight_5 :")
        print(delta_weight_5)


        delta_bias_5 = (a * (-1) * grad_5).T
        print("delta_bias_5 :")
        print(delta_bias_5)

        print("---------------------------")

        grad_34 = gradient(y34.T, grad_5*self.weight_5)

        print("---------------------------")


        delta_weight_34 = (a * train_data * grad_34).T
        print("delta_weight_34 :")
        print(delta_weight_34)

        delta_bias_34 = (a * (-1) * grad_34).T
        print("delta_bias_34 :")
        print(delta_bias_34)
        print("---------------------------")


        print("-------------update ------------")
        self.weight_5    += delta_weight_5
        self.weight_34   += delta_weight_34
        self.bias_5      += delta_bias_5
        self.bias_34     += delta_bias_34

        print("weight_34 :")
        print(self.weight_34)
        print("weight_5 :")
        print(self.weight_5)
        print("bias_34 :")
        print(self.bias_34)
        print("bias_5 :")
        print(self.bias_5)

        return final_err

    def printParam(self):
        print("======={0}======".format(self.name))

        for layer in self.layers:
            if type(layer) != SingleLayer:
                # prevLayer 중복지정 방지때문에 if then else로 처리
                pass
            else:
                print("-------{0}-------".format(layer.name))
                print(layer.weight)
                print(layer.bias)
                print("")




mlp_xor = MLP(2,2,1, "xor")
mlp_and = MLP(2,2,1, "and")
mlp_or = MLP(2,2,1, "or")



train_data_xor = np.array([[[1], [1]],[[1],[0]], [[0],[1]], [[0],[0]]])
supervised_xor = np.array([[0], [1], [1], [0]])

train_data_and = np.array([[[1], [1]],[[1],[0]], [[0],[1]], [[0],[0]]])
supervised_and = np.array([[1], [0], [0], [0]])

train_data_or = np.array([[[1], [1]],[[1],[0]], [[0],[1]], [[0],[0]]])
supervised_or = np.array([[1], [1], [1], [0]])

errListXor = []
errListAnd = []
errListOr = []
errIndex = []

tr_rand_xor_test = copy.deepcopy(train_data_xor)
tr_rand_xor_supervise = copy.deepcopy(supervised_xor)

tr_rand_and_test = copy.deepcopy(train_data_and)
tr_rand_and_supervise = copy.deepcopy(supervised_and)

tr_rand_or_test = copy.deepcopy(train_data_or)
tr_rand_or_supervise = copy.deepcopy(supervised_or)






for i in range(20000):
    epoch_shuffle = np.arange(4)
    np.random.shuffle(epoch_shuffle)
    for idx in epoch_shuffle:
         mlp_xor.train(train_data_xor[idx], supervised_xor[idx])
         mlp_and.train(train_data_and[idx], supervised_and[idx])
         mlp_or.train(train_data_or[idx], supervised_or[idx])

    #ranNum = np.random.randint(0,3+1)

    #mlp_xor.train(train_data_xor[ranNum], supervised_xor[ranNum])
    #mlp_and.train(train_data_and[ranNum], supervised_and[ranNum])
    #mlp_or.train(train_data_or[ranNum], supervised_or[ranNum])

    if i%100==0:
        tmpErrXor = []
        tmpErrAnd = []
        tmpErrOr = []

        for j in range(len(tr_rand_xor_test)):
            err = mlp_xor.errorRate(tr_rand_xor_test[j], tr_rand_xor_supervise[j])
            tmpErrXor.append(err)

            err = mlp_and.errorRate(tr_rand_and_test[j], tr_rand_and_supervise[j])
            tmpErrAnd.append(err)

            err = mlp_or.errorRate(tr_rand_or_test[j], tr_rand_or_supervise[j])
            tmpErrOr.append(err)

        errListXor.append( sum([ x**2 for x in tmpErrXor ]).item() )
        errListAnd.append( sum([ x**2 for x in tmpErrAnd ]).item() )
        errListOr.append( sum([ x**2 for x in tmpErrOr ]).item() )
        errIndex.append(i)


    #err = mlp.train(train_data[0], supervised[0])



"""
errListXor = []
errListAnd = []
errListOr = []

tr_rand_xor_test = copy.deepcopy(train_data_xor)
tr_rand_xor_supervise = copy.deepcopy(supervised_xor)

tr_rand_and_test = copy.deepcopy(train_data_and)
tr_rand_and_supervise = copy.deepcopy(supervised_and)

tr_rand_or_test = copy.deepcopy(train_data_or)
tr_rand_or_supervise = copy.deepcopy(supervised_or)

for i in range(len(tr_rand_xor_test)):
    err = mlp_xor.errorRate(tr_rand_xor_test[i], tr_rand_xor_supervise[i])
    errListXor.append(err)

    err = mlp_and.errorRate(tr_rand_and_test[i], tr_rand_and_supervise[i])
    errListAnd.append(err)

    err = mlp_or.errorRate(tr_rand_or_test[i], tr_rand_or_supervise[i])
    errListOr.append(err)

errorSumOfPoweredXor = sum( [ x**2 for x in errListXor ] )
errorSumOfPoweredAnd = sum( [ x**2 for x in errListAnd ] )
errorSumOfPoweredOr = sum( [ x**2 for x in errListOr ] )

print("Xor 에러제곱합 : ", end="")
print(errorSumOfPoweredXor)
print("And 에러제곱합 : ", end="")
print(errorSumOfPoweredAnd)
print("Or 에러제곱합 : ", end="")
print(errorSumOfPoweredOr)
errListXor.clear(); errListAnd.clear(); errListOr.clear()
"""

print("=============================")
print(mlp_xor.test(np.array( [[0],[0]] )))
print(mlp_xor.test(np.array( [[0],[1]] )))
print(mlp_xor.test(np.array( [[1],[0]] )))
print(mlp_xor.test(np.array( [[1],[1]] )))
print("=============================")
print(mlp_and.test(np.array( [[0],[0]] )))
print(mlp_and.test(np.array( [[0],[1]] )))
print(mlp_and.test(np.array( [[1],[0]] )))
print(mlp_and.test(np.array( [[1],[1]] )))
print("=============================")
print(mlp_or.test(np.array( [[0],[0]] )))
print(mlp_or.test(np.array( [[0],[1]] )))
print(mlp_or.test(np.array( [[1],[0]] )))
print(mlp_or.test(np.array( [[1],[1]] )))
print("=============================")

# (w, h, index)
fig = plt.figure()
fig.suptitle("Multi Layered Perceptron boolean classfier")
ay1 = fig.add_subplot(1,3,1)
ay2 = fig.add_subplot(1,3,2)
ay3 = fig.add_subplot(1,3,3)
ay1.plot(errIndex, errListXor, label="y = sum of error_rate**2")
ay2.plot(errIndex, errListAnd, label="y = sum of error_rate**2")
ay3.plot(errIndex, errListOr, label="y = sum of error_rate**2")
plt.title("Showing reduced error rate")
#fig.show()
plt.show()

# plt.plot(errIndex, errListXor)
# plt.plot(errIndex, errListAnd)
# plt.plot(errIndex, errListOr)


""" 각 신경망의 파라미터들 출력 """
mlp_xor.printParam()
mlp_and.printParam()
mlp_or.printParam()

from matplotlib import pyplot as plt
import numpy as np

#(w1 * x1) + (w2 * x2) - t


""" 훈련된 XOR 입력 패턴 분류 신경망 중 하나의 파라미터로 도출된 2차원 도면 표현"""
w1_1 = 4.12375411
w2_1 = 4.11868558
t_1 = 6.31525069

x_1 = np.arange(-5,5+1)
y_1 = - (w1_1 * x_1 - t_1) / w2_1



w1_2 = 6.20146275
w2_2 = 6.17677067
t_2 = 2.66214285


x_2 = np.arange(-5,5+1)
y_2 = - (w1_2 * x_2 - t_2) / w2_2

# x1, y1, x2, y2, horizontal_line, vertical_line
plt.plot(x_1, y_1, "r-" , x_2, y_2, "b-", [0,0], [-5,5], "k-" ,[-5,5] ,[0,0], "k-")