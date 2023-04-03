import numpy

# Problem 1
def euler(a, b, N, init, f):
    h = (b - a) / N
    t = a
    w = init
    for i in range(1, N+1):
        w += h*f(t, w)
        t = a + i*h
    return w

def f(t, y): 
    return t - y ** 2

print("%.5f\n" %  euler(0, 2, 10, 1, f))

def rungeKutta(a, b, N, init, f):
    h = (b - a) / N
    t = a
    w = init
    for i in range(1, N+1):
        K1 = h * f(t, w)
        K2 = h * f(t + h / 2, w + K1 / 2)
        K3 = h * f(t + h / 2, w + K2 / 2)
        K4 = h * f(t + h, w + K3)
        w = w + (K1 + 2*K2 + 2*K3 + K4) / 6
        t = a + i * h
    return w

print("%.5f\n" % rungeKutta(0, 2, 10, 1, f))

def gaussian(n, A):
    for i in range(0, n-1):
        p = -1
        for j in range(i, n):
            if not A[j, i] == 0:
                p = j
                break
        A[i], A[p] = A[p], A[i]
        for j in range(i+1, n):
            m = A[j, i] / A[i, i]
            for k in range(0, n+1):
                A[j, k] -= m * A[i, k]
    if A[n-1, n-1] == 0:
        return numpy.matrix([])
    x = numpy.zeros(n)
    x[n-1] = A[n-1, n] / A[n-1, n-1]
    for i in range(n-2, -1, -1):
        x[i] = A[i, n]
        for j in range(i+1, n):
            x[i] -= A[i, j] * x[j]
        x[i] /= A[i, i]
    return x

A = numpy.matrix('2 -1 1 6; 1 3 1 0; -1 5 4 -3')
print(gaussian(3, A))
print()

def LUDecomp(n, A):
    
    L = numpy.zeros((n, n))
    U = numpy.zeros((n, n))

    L[0, 0] = 1
    U[0, 0] = A[0, 0]

    for j in range(1, n):
        U[0, j] = A[0, j] / L[0, 0]
        L[j, 0] = A[j, 0] / U[0, 0]
    
    for i in range(1, n):
        L[i, i] = 1
        U[i, i] = A[i, i]
        for k in range(0, i):
            U[i, i] -= L[i, k] * U[k, i]
        
        for j in range(i+1, n):
            U[i, j] = A[i, j]
            L[j, i] = A[j, i] 
            for k in range(0, i):
                U[i, j] -= L[i, k] * U[k, j]
                L[j, i] -= L[j, k] * U[k, i]
            U[i, j] /= L[i][i]
            L[j, i] /= U[i][i]
    
    det = 1.0
    for i in range(n):
        det *= U[i, i]

    print("%.5f\n" % det) 
    print()

    print(L)
    print()

    print(U)
    print()

A = numpy.matrix('1 1 0 3; 2 1 -1 1; 3 -1 -1 2; -1 2 3 -1')
LUDecomp(4, A)     

def isDiagonallyDominant(n, A):
    for i in range(0, n):
        tot = 0
        for j in range(0, n):
            if not i == j:
                tot += A[i, j]
        if(tot > A[i, i]):
            return False
    return True

A = numpy.matrix('9 0 5 2 1; 3 9 1 2 1; 0 1 7 2 3; 4 2 3 12 2; 3 2 4 0 8')
print(isDiagonallyDominant(5, A))
print()

def isPositiveDefinite(n, A):
    L = numpy.zeros((n, n))
    D = numpy.zeros(n)

    for i in range(n):
        v = numpy.zeros(n)
        for j in range(i):
            v[j] = L[i, j] * D[j]
        D[i] = A[i, i]
        for j in range(i):
            D[i] -= L[i, j] * v[j]
        for j in range(i+1, n):
            L[j, i] = A[j, i]
            for k in range(i):
                L[j, i] -= L[j, k] * v[k]
            L[j, i] /= D[i]
    
    good = True
    for i in range(n):
        good = good and D[i] > 0
    
    return good

print(isPositiveDefinite(3, numpy.matrix('2 2 1; 2 3 0; 1 0 2')))