import numpy as np
import matplotlib.pyplot as plt

def model(w, b):
    def model_fun(x):
        return w * x + b
    return model_fun

def cost_fun(w, b, x_data, y_data):
    sum = 0
    m = x_data.shape[0]
    f = model(w,b)
    for i in range(m):
        sum += (f(x_data[i]) - y_data[i]) ** 2
    return sum/(2*m)

def display_model_w_data(f, x_data, y_data):
    xs = np.linspace(min(x_data), max(x_data))
    ys = f(xs)
    plt.plot(x_data, y_data, 'o', color='r', label='Data', )
    plt.plot(xs, ys, color='b', label='Model')
    plt.legend()
    plt.show()  


def grad_desc(x_data, y_data, alpha, iters):
    w,b = 0,0
    f = model(w,b)
    m = x_data.shape[0]

    for _ in range(iters):

        cost_dw = 0
        for i in range(m):
            cost_dw += (f(x_data[i]) - y_data[i]) * x_data[i]
        cost_dw /= m

        cost_db = 0
        for i in range(m):
            cost_db += (f(x_data[i]) - y_data[i])
        cost_db /= m

        w -= alpha * cost_dw
        b -= alpha * cost_db

        f = model(w, b)

    return f