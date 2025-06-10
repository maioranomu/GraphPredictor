import random
import numpy
import math

result = []

def graphs():
    result = []

    # Exponential growth (with controlled scale)
    for _ in range(100):
        nums = []
        weight = random.randint(2, 5)
        for i in range(10):
            nums.append(weight ** (i + 1))  # Exponential
        result.append(nums)

    # Linear patterns with random slope and intercept
    for _ in range(250):
        nums = []
        weight = random.uniform(0.5, 3.0)
        intercept = random.uniform(-5, 5)
        for i in range(1, 11):
            nums.append(round(weight * i + intercept, 2))
        result.append(nums)

    # Multiples (linear with zero intercept)
    for _ in range(250):
        nums = []
        factor = random.randint(1, 10)
        for i in range(1, 11):
            nums.append(factor * i)
        result.append(nums)

    # Sine wave with low noise
    for _ in range(250):
        nums = []
        amplitude = random.uniform(1.0, 3.0)
        frequency = random.uniform(0.2, 0.6)
        phase = random.uniform(0, 2 * math.pi)
        offset = random.uniform(-2, 2)
        noise_scale = random.uniform(0, 0.3)

        for i in range(1, 11):
            noise = random.uniform(-noise_scale, noise_scale)
            val = amplitude * math.sin(frequency * i + phase) + offset + noise
            nums.append(round(val, 2))
        result.append(nums)

    # Quadratic sequences
    for _ in range(250):
        nums = []
        a = random.uniform(0.2, 1.5)
        b = random.uniform(-3, 3)
        c = random.uniform(-5, 5)
        noise_scale = random.uniform(0, 1)

        for i in range(1, 11):
            noise = random.uniform(-noise_scale, noise_scale)
            val = a * i**2 + b * i + c + noise
            nums.append(round(val, 2))
        result.append(nums)

    # Exponential decay/growth with small noise
    for _ in range(250):
        nums = []
        base = random.uniform(1.05, 1.3)
        scale = random.uniform(1, 2)
        offset = random.uniform(-5, 5)
        noise_scale = random.uniform(0, 0.5)

        for i in range(1, 11):
            noise = random.uniform(-noise_scale, noise_scale)
            val = scale * (base ** i) + offset + noise
            nums.append(round(val, 2))
        result.append(nums)

    # Cubic sequences
    for _ in range(250):
        nums = []
        a = random.uniform(-0.05, 0.05)
        b = random.uniform(-0.5, 0.5)
        c = random.uniform(-2, 2)
        d = random.uniform(-5, 5)
        noise_scale = random.uniform(0, 0.5)

        for i in range(1, 11):
            noise = random.uniform(-noise_scale, noise_scale)
            val = a * i**3 + b * i**2 + c * i + d + noise
            nums.append(round(val, 2))
        result.append(nums)

    for i in result:
        print(i)

def main():
    graphs()
    random.shuffle(result)
    export = []
    for row in result:
        export.append(', '.join(map(str, row)))
        print(', '.join(map(str, row)))

    should_rewrite = input("REWRITE CURRENT DATA? [Y/N]")
    if should_rewrite.lower() == "y":
        with open("data.csv", "w") as file:
            file.write("n1,n2,n3,n4,n5,n6,n7,n8,n9,n10\n")
            for item in export:
                file.write(f"{item}\n")
main()
