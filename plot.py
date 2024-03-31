import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


def anim_gen(f, X_sgd, Y_sgd, X_m, Y_m, X_rms, Y_rms, X_adag, Y_adag, X_adad, Y_adad, X_adam, Y_adam, X_nag, Y_nag):
    X, Y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))  # Rosenbrock function
    # X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))  # Styblinski–Tang function
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    ax1.contour(X, Y, f(X, Y), 100, cmap='jet')
    ax1.scatter(1, 1, marker='^', color='black', alpha=0.5, s=60)  # Min of Rosenbrock function
    # ax1.scatter(-2.903534, -2.903534, marker='^', color='black', alpha=0.5, s=60)  # Min of Styblinski–Tang function
    # Create animation
    line_sgd, = ax1.plot([], [], 'r', label='SGD', lw=2.0, alpha=0.5)
    point_sgd, = ax1.plot([], [], 'o', color='red', markeredgecolor='black', markersize=10)
    line_m, = ax1.plot([], [], 'g', label='SGD + Momentum', lw=2.0, alpha=0.5)
    point_m, = ax1.plot([], [], 'o', color='green', markeredgecolor='black', markersize=10)
    line_rms, = ax1.plot([], [], 'b', label='RMSProp', lw=2.0, alpha=0.5)
    point_rms, = ax1.plot([], [], 'o', color='blue', markeredgecolor='black', markersize=10)
    line_adag, = ax1.plot([], [], 'm', label='Adagrad', lw=2.0, alpha=0.5)
    point_adag, = ax1.plot([], [], 'o', color='magenta', markeredgecolor='black', markersize=10)
    line_adad, = ax1.plot([], [], 'y', label='Adadelta', lw=2.0, alpha=0.5)
    point_adad, = ax1.plot([], [], 'o', color='yellow', markeredgecolor='black', markersize=10)
    line_adam, = ax1.plot([], [], 'k', label='Adam', lw=2.0, alpha=0.5)
    point_adam, = ax1.plot([], [], 'o', color='black', markeredgecolor='black', markersize=10)
    line_nag, = ax1.plot([], [], 'c', label='NAG', lw=2.0, alpha=0.5)
    point_nag, = ax1.plot([], [], 'o', color='cyan', markeredgecolor='black', markersize=10)

    def init_1():
        line_sgd.set_data([], [])
        point_sgd.set_data([], [])
        line_m.set_data([], [])
        point_m.set_data([], [])
        line_rms.set_data([], [])
        point_rms.set_data([], [])
        line_adag.set_data([], [])
        point_adag.set_data([], [])
        line_adad.set_data([], [])
        point_adad.set_data([], [])
        line_adam.set_data([], [])
        point_adam.set_data([], [])
        line_nag.set_data([], [])
        point_nag.set_data([], [])
        return line_sgd, point_sgd, line_m, point_m, line_rms, point_rms, line_adag, point_adag, \
            line_adad, point_adad, line_adam, point_adam, line_nag, point_nag

    def animate_1(i):
        # Animate line
        line_sgd.set_data(X_sgd[:i], Y_sgd[:i])
        line_m.set_data(X_m[:i], Y_m[:i])
        line_rms.set_data(X_rms[:i], Y_rms[:i])
        line_adag.set_data(X_adag[:i], Y_adag[:i])
        line_adad.set_data(X_adad[:i], Y_adad[:i])
        line_adam.set_data(X_adam[:i], Y_adam[:i])
        line_nag.set_data(X_nag[:i], Y_nag[:i])
        # Animate points
        point_sgd.set_data(X_sgd[i], Y_sgd[i])
        point_m.set_data(X_m[i], Y_m[i])
        point_rms.set_data(X_rms[i], Y_rms[i])
        point_adag.set_data(X_adag[i], Y_adag[i])
        point_adad.set_data(X_adad[i], Y_adad[i])
        point_adam.set_data(X_adam[i], Y_adam[i])
        point_nag.set_data(X_nag[i], Y_nag[i])
        return line_sgd, point_sgd, line_m, point_m, line_rms, point_rms, line_adag, point_adag, \
            line_adad, point_adad, line_adam, point_adam, line_nag, point_nag

    ax1.legend(loc=2)
    anim1 = animation.FuncAnimation(fig1, animate_1, init_func=init_1,
                                    frames=len(X_sgd), interval=100,
                                    repeat_delay=60, blit=True)
    mywriter = animation.FFMpegWriter(fps=60)
    anim1.save('anim1.mp4', writer=mywriter)