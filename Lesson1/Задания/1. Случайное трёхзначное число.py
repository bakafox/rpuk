import random

num = random.randint(100, 999)
print(num)

sum = 0
for digit in range(0, len(str(num))):
    sum += num % 10
    num //= 10
print(sum)