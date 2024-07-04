import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y_train = np.array([0, 0, 0, 1, 1, 1])

w_tmp = np.array([2.,3.])
b_tmp = 0.

pos=(y_train==1)
neg=(y_train==0)
x1_feature=np.array([x[0] for x in X_train])
x2_feature=np.array([x[1] for x in X_train])
'''
plt.scatter(x1_feature[pos], x2_feature[pos], marker="x", c="red")
plt.scatter(x1_feature[neg], x2_feature[neg], marker="o", c="blue")
plt.show()
'''
def model_output(x, w, b):
    m=x.shape[0]
    f_wb=np.zeros(m)
    for i in range(m):
        f_wb[i]=1/(1+(np.exp(-(np.dot(w,x[i])+b))))
    return f_wb

def compute_cost(x, y, w, b):
    m=x.shape[0]
    tmp_fwb=model_output(x, w, b)
    cost=0
    for i in range(m):
        cost+= (-y[i]*np.log(tmp_fwb[i]))-((1-y[i])*np.log(1-tmp_fwb[i]))
    return cost/m

def compute_gradient(x, y, w, b):
    n=w.shape[0]
    m=x.shape[0]
    tmp_gradient=np.zeros(n)
    tmp_output=model_output(x, w, b)
    tmp_db=0
    for j in range(n):
        calc=0
        for i in range(m):
            calc+=((tmp_output[i]-y[i])*x[i][j])
        tmp_gradient[j]=calc/m
    calc=0
    for i in range(m):
        calc+=tmp_output[i]-y[i]
    tmp_db=calc/m
    return  tmp_gradient,tmp_db

def gradient_descent(x, y, w, b, alpha, iterations):
    w_tmp,b_tmp=w,b
    for i in range(iterations):
        tmpdw_g,tmpdb_g=compute_gradient(x, y, w_tmp, b_tmp)
        w_tmp=w_tmp-(alpha*tmpdw_g)
        b_tmp=b_tmp-(alpha*tmpdb_g)
        if((i%1000)==0):
            print("Cost in iteration number",i,"is:", compute_cost(x, y, w_tmp, b_tmp))
    print("Final Cost: ",compute_cost(x, y, w_tmp, b_tmp))
    return w_tmp, b_tmp

tmp_alpha = 10
iterations = 1000
final_w,final_b= gradient_descent(X_train, y_train, w_tmp, b_tmp, tmp_alpha, iterations)
print(final_w,final_b)