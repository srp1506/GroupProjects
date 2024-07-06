test_cases = int(input())

while test_cases != 0:
    
    test_cases -= 1
    val = int(input())
    
    for number in range(val):
        
        if number == 0:
            number += 3
        
        elif number % 2 == 0:
            number *= 2
        
        elif number % 2 != 0:
            number *= number            
            
        print(number, end = " ")
    
    print()