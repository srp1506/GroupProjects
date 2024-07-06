test_cases = int(input())

while test_cases != 0:
    
    test_cases -= 1
    row_number = int(input())
    
    for char_number in reversed(range(1, row_number + 1)):
        
        for index in range(1, char_number + 1):
            
            if index % 5 == 0:
                print("#",end = "")
            else:
                print("*",end = "")
        
        print("")