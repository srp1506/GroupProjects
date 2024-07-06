test_cases = int(input())

while test_cases != 0:
    
    test_cases -= 1
    user_input = input().lower()
    length = len(user_input)
    
    if (length % 2 != 0):
        limit = int((length - 1) / 2)
    else:
        limit = int(length / 2)

    for i in range(limit):
        
        if (user_input[i] != user_input[length - i - 1]):
            
            print("It is not a palindrome")
            break
    else:
        print("It is a palindrome")
    