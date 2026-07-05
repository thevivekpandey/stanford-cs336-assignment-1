import ipdb

def factorial(n):
    if n <= 0:
        return 1
    else:
        return n * factorial(n - 1)

ipdb.set_trace()
num = 5
fact = factorial(num)

print("Factorial of ", num, "is", fact)
