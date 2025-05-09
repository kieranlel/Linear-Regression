import numpy as np
import matplotlib.pyplot as plt

def model(w, b):

    def model_fun(x):
        return np.dot(w,x) + b
    
    return model_fun

def cost_fun(w, b, x_data, y_data):

    sum = 0
    m = x_data.shape[0]
    f = model(w,b)

    for i in range(m):
        sum += (f(x_data[i]) - y_data[i]) ** 2
    return sum/(2*m)

def display_model_w_data(f, x_data, y_data, n):

    if n == 1:
        f_out = [f(x) for x in x_data]
        # xs = np.linspace(min(x_data), max(x_data))
        # ys = f(xs)
        plt.plot(x_data, y_data, 'o', color='r', label='Data', )
        plt.plot(x_data, f_out, color='b', label='Model')
        plt.legend()
        plt.show()  

    elif n == 2:
        # 3D graph here
        pass
        


def grad_desc(x_data, y_data, alpha, iters):
    w = np.array([0] * len(x_data[0]), dtype='float64')
    b = 0
    f = model(w,b)
    m = len(x_data)

    for _ in range(iters):

        # Calculate gradient wrt w
        cost_dw = np.array([0] * len(x_data[0]))
        cost_dw = cost_dw.astype('float64')
        for i in range(len(x_data[0])):
            sum = 0
            for j in range(m):
                sum += (f(x_data[j]) - y_data[j]) * x_data[j,i]
            sum /= m
            cost_dw[i] = sum

        # Calculate gradient wrt b
        cost_db = 0
        for i in range(m):
            cost_db += (f(x_data[i]) - y_data[i])
        cost_db /= m

        w -= alpha * cost_dw
        b -= alpha * cost_db

        f = model(w, b)
        
    return f