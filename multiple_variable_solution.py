import numpy as np

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
Y_train = np.array([460, 232, 178])

w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
b_init = 785.1811367994083

def model_output(x, w, b):
    m=x.shape[0]
    f_wb=np.zeros(m)
    for i in range(m):
        f_wb[i]=np.dot(w, x[i])+b
    return f_wb

def compute_cost(model_result, x, y, w, b):
    m = x.shape[0]
    tmp_fwb = model_result(x, w, b)
    total_cost = 0
    for i in range(m):
        total_cost += (tmp_fwb[i] - y[i]) ** 2
    total_cost = total_cost / (2 * m)
    return total_cost

def compute_gradient(model_result, x, y, w, b):
    n=w.shape[0]
    m=x.shape[0]
    tmp_gradient=np.zeros(n)
    tmp_output=model_result(x, w, b)
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

def gradient_descent(model_result, gradient, cost, w, b, x, y, alpha, iterations):
    w_tmp,b_tmp=w,b
    for i in range(iterations):
        tmpdw_g,tmpdb_g=gradient(model_result, x, y, w_tmp, b_tmp)
        w_tmp=w_tmp-(alpha*tmpdw_g)
        b_tmp=b_tmp-(alpha*tmpdb_g)
        if((i%100)==0):
            print("Cost in iteration number",i+1,"is:", cost(model_result, x, y, w_tmp, b_tmp))
    return w_tmp, b_tmp

tmp_alpha = 1.0e-7
iterations = 6000

final_w,final_b= gradient_descent(model_output, compute_gradient, compute_cost, w_init, b_init, X_train, Y_train, tmp_alpha, iterations)
