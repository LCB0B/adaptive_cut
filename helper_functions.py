def swap(a,b):
    if a > b:
        return b,a
    return a,b

def Dc(m, n):
    try:
        return (m * (m - n + 1.0)) / ((n - 2.0) * (n - 1.0))
    except ZeroDivisionError:
        return 0.0

def Dc2(m, n):
    try:
        return ( (m - n + 1.0)) / (n*(n - 1.0)/2 -(n - 1.0))
    except ZeroDivisionError:
        return 0.0
    
def two_among(n):
    return n*(n-1)//2