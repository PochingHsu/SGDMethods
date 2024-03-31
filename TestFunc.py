# Rosenbrock function
def rosenbrock(x,y):
    a = 1
    b = 100
    rb = ((a - x)**2) + (b*((y - x**2)**2))
    return rb


# The derivative of Rosenbrock function
def d_rosenbrock(x,y):
    a = 1
    b = 100
    drb_dx = -2*a*(1 - x) - 4*b*x*(y - (x**2))
    drb_dy = 2*b*(y - (x**2))
    return drb_dx, drb_dy


# Styblinski–Tang function
def styblinski_tang(x,y):
    st = (x**4-16*x**2+5*x+y**4-16*y**2+5*y)/2
    return st


# The derivative of Styblinski–Tang function
def d_styblinski_tang(x,y):
    dst_dx = (4*x**3-32*x+5)/2
    dst_dy = (4*y**3-32*y+5)/2
    return dst_dx, dst_dy
