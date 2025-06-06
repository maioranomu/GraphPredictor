import random
import numpy as np

result = []

for _ in range(1000):
    nums = []
    a = random.uniform(0.5, 2.0)      # Quadratic coefficient
    b = random.uniform(-5, 5)         # Linear coefficient
    c = random.uniform(-10, 10)       # Constant
    noise_scale = random.uniform(0, 3)

    for i in range(1, 11):
        noise = random.uniform(-noise_scale, noise_scale)
        val = a * (i ** 2) + b * i + c + noise
        nums.append(round(val, 2))    # Rounded for readability

    result.append(nums)

# Print as CSV rows
for row in result:
    print(', '.join(map(str, row)))
