from SGDs import SGDs
from TestFunc import rosenbrock, d_rosenbrock, styblinski_tang, d_styblinski_tang
from plot import anim_gen

# Test function
f = rosenbrock
df = d_rosenbrock
# f = styblinski_tang
# df = d_styblinski_tang

# Setting and initialization
xint, yint = -0.1, 0.2
lr = 0.001
epoch = 2000
sg = SGDs(f=f, df=df, xint=xint, yint=yint, epoch=epoch, lr=lr)

# Vanilla SGD
X_sgd, Y_sgd = sg.sgd_vanilla()
# SGD+Momentum
X_m, Y_m = sg.sgd_momentum(m=0.5)
# RMSProp
decay = 0.9
X_rms, Y_rms = sg.rmsprop(decay)
# NAG
X_nag, Y_nag = sg.nag(m=0.3)
# Adagrad
X_adag, Y_adag = sg.adagrad()
# Adadelta
X_adad, Y_adad = sg.adadelta(rho=0.99)
# Adam
X_adam, Y_adam = sg.adam(beta1=0.8, beta2=0.999)
# Generate animation
anim_gen(f, X_sgd, Y_sgd, X_m, Y_m, X_rms, Y_rms, X_adag, Y_adag, X_adad, Y_adad, X_adam, Y_adam, X_nag, Y_nag)

