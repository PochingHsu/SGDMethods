import numpy as np
import math


class SGDs:
    def __init__(self, f, df, xint, yint, epoch, lr):
        self.f = f
        self.df = df
        self.xint = xint
        self.yint = yint
        self.epoch = epoch
        self.lr = lr
    def sgd_vanilla(self):
        # initialization
        x = self.xint
        y = self.yint
        # log variables & function value
        x_sgd, y_sgd = [], []
        f_sgd = np.zeros(self.epoch)
        for i in range(self.epoch):
            f_sgd[i] = self.f(x,y)
            df_sgd = np.array(self.df(x, y))
            x = x - np.dot(self.lr, df_sgd[0])
            y = y - np.dot(self.lr, df_sgd[1])
            x_sgd.append(x)
            y_sgd.append(y)
            print('Epoch [{:10}/{}], function value: {:.10f}, x:{:.10f}, y:{:.10f}'.format(i + 1, self.epoch, f_sgd[i], x, y))
        return x_sgd, y_sgd

    def sgd_momentum(self, m):
        # initialization
        x = self.xint
        y = self.yint
        u = 0
        v = 0
        # log variables & function value
        x_sgd, y_sgd = [], []
        f_sgd = np.zeros(self.epoch)
        for i in range(self.epoch):
            f_sgd[i] = self.f(x,y)
            df_sgd = np.array(self.df(x, y))
            u = m * u - self.lr * df_sgd[0]
            v = m * v - self.lr * df_sgd[1]
            x = x + u
            y = y + v
            x_sgd.append(x)
            y_sgd.append(y)
            print('Epoch [{:10}/{}], function value: {:.10f}, x:{:.10f}, y:{:.10f}'.format(i + 1, self.epoch, f_sgd[i], x, y))
        return x_sgd, y_sgd

    def rmsprop(self, decay, eps=1e-8):
        # initialization
        x = self.xint
        y = self.yint
        cache_x = 0
        cache_y = 0
        # log variables & function value
        x_sgd, y_sgd = [], []
        f_sgd = np.zeros(self.epoch)
        for i in range(self.epoch):
            f_sgd[i] = self.f(x,y)
            df_sgd = np.array(self.df(x, y))
            cache_x = decay * cache_x + (1 - decay) * df_sgd[0] ** 2
            cache_y = decay * cache_y + (1 - decay) * df_sgd[1] ** 2
            x = x - self.lr * df_sgd[0] / (math.sqrt(cache_x + eps))
            y = y - self.lr * df_sgd[1] / (math.sqrt(cache_y + eps))
            x_sgd.append(x)
            y_sgd.append(y)
            print('Epoch [{:10}/{}], function value: {:.10f}, x:{:.10f}, y:{:.10f}'.format(i + 1, self.epoch, f_sgd[i], x, y))
        return x_sgd, y_sgd

    def adagrad(self):
        # initialization
        x = self.xint
        y = self.yint
        # log variables & function value
        x_sgd, y_sgd = [], []
        f_sgd = np.zeros(self.epoch)
        sg = np.zeros(2)
        lr_adag = np.zeros(2)
        for i in range(self.epoch):
            f_sgd[i] = self.f(x,y)
            df_sgd = np.array(self.df(x, y))
            for j in range(df_sgd.shape[0]):
                sg[j] += df_sgd[j] ** 2.0
                lr_adag[j] = self.lr / (1e-8 + math.sqrt(sg[j]))
            x = x - np.dot(lr_adag[0], df_sgd[0])
            y = y - np.dot(lr_adag[1], df_sgd[1])
            x_sgd.append(x)
            y_sgd.append(y)
            print('Epoch [{:10}/{}], function value: {:.10f}, x:{:.10f}, y:{:.10f}'.format(i + 1, self.epoch, f_sgd[i], x, y))
        return x_sgd, y_sgd

    def adadelta(self, rho, eps=1e-4):
        # initialization
        x = self.xint
        y = self.yint
        sg = np.zeros(2)
        sg_mv = np.zeros(2)
        sd_mv = np.zeros(2)
        lr_adad = np.zeros(2)
        # log variables & function value
        x_sgd, y_sgd = [], []
        f_sgd = np.zeros(self.epoch)
        for i in range(self.epoch):
            f_sgd[i] = self.f(x,y)
            df_sgd = np.array(self.df(x, y))
            for j in range(df_sgd.shape[0]):
                sg[j] = df_sgd[j] ** 2.0
                sg_mv[j] = (sg_mv[j] * rho) + (sg[j] * (1.0 - rho))
                lr_adad[j] = (eps + math.sqrt(sd_mv[j])) / (eps + math.sqrt(sg_mv[j]))
                sd_mv[j] = (sd_mv[j] * rho) + ((lr_adad[j] * df_sgd[j]) ** 2.0 * (1.0 - rho))
            x = x - lr_adad[0] * df_sgd[0]
            y = y - lr_adad[1] * df_sgd[1]
            x_sgd.append(x)
            y_sgd.append(y)
            print('Epoch [{:10}/{}], function value: {:.10f}, x:{:.10f}, y:{:.10f}'.format(i + 1, self.epoch, f_sgd[i], x, y))
        return x_sgd, y_sgd

    def adam(self, beta1, beta2, eps=1e-8):
        # initialization
        x = self.xint
        y = self.yint
        # log variables & function value
        x_sgd, y_sgd = [], []
        f_sgd = np.zeros(self.epoch)
        m = np.zeros(2)
        mhat = np.zeros(2)
        v = np.zeros(2)
        vhat = np.zeros(2)
        for i in range(self.epoch):
            f_sgd[i] = self.f(x,y)
            df_sgd = np.array(self.df(x, y))
            for j in range(df_sgd.shape[0]):
                m[j] = beta1 * m[j] + (1.0 - beta1) * df_sgd[j]
                v[j] = beta2 * v[j] + (1.0 - beta2) * df_sgd[j] ** 2
                mhat[j] = m[j] / (1.0 - beta1 ** (i + 1))
                vhat[j] = v[j] / (1.0 - beta2 ** (i + 1))
            x = x - np.dot(self.lr, (mhat[0] / (math.sqrt(vhat[0]) + eps)))
            y = y - np.dot(self.lr, (mhat[1] / (math.sqrt(vhat[1]) + eps)))
            x_sgd.append(x)
            y_sgd.append(y)
            print('Epoch [{:10}/{}], function value: {:.10f}, x:{:.10f}, y:{:.10f}'.format(i + 1, self.epoch, f_sgd[i], x, y))
        return x_sgd, y_sgd

    def nag(self, m):
        # initialization
        x = self.xint
        y = self.yint
        # log variables & function value
        x_sgd, y_sgd = [], []
        f_sgd = np.zeros(self.epoch)
        delta = np.zeros(2)
        for i in range(self.epoch):
            f_sgd[i] = self.f(x,y)
            xhat = x + m * delta[0]
            yhat = y + m * delta[1]
            df_sgd = np.array(self.df(xhat, yhat))
            for j in range(df_sgd.shape[0]):
                delta[j] = (m * delta[j]) - self.lr * df_sgd[j]
            x = x + delta[0]
            y = y + delta[1]
            x_sgd.append(x)
            y_sgd.append(y)
            print('Epoch [{:10}/{}], function value: {:.10f}, x:{:.10f}, y:{:.10f}'.format(i + 1, self.epoch, f_sgd[i], x, y))
        return x_sgd, y_sgd