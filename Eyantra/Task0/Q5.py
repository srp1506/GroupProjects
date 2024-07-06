test_cases = int(input())

while test_cases != 0:
    
    test_cases -= 1
    inputs = input().split("@")
    
    for i in range(len(inputs)):
        
        sentence = inputs[i].split()
        
        for j in range(len(sentence)):
            
            print(len(sentence[j]),end = "")
            if j != (len(sentence) - 1):
                print(end = ",")
            
        print()