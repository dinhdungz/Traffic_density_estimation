import math
mean = 0
var = 1
x = 0
p = math.exp((-(x - mean)**2)/(2*var))/(math.sqrt(2*math.pi*var))
print(p)