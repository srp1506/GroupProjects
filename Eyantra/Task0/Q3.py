import math as m

test_cases = int(input())

while test_cases != 0:
    
    test_cases -= 1
    inputs = input().split() 
    x1 = int(inputs[0])
    y1 = int(inputs[1])
    x2 = int(inputs[2])
    y2 = int(inputs[3])
    deltax = x2 - x1
    deltay = y2 - y1
    distance = m.pow((deltax * deltax) + (deltay * deltay), 0.5)
    print("Distance: %.2f" % distance)