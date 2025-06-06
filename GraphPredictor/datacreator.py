import random
import numpy
import math

result = []

def graphs():
    for _ in range(250):
        nums = []
        weight = random.randint(1, 10)
        weight2 = random.randint(1, 10)
        for i in range(1, 11):
            nums.append(weight*i+weight2)
        result.append(nums)
        
    for _ in range(250):
        nums = []
        weight = random.randint(1, 10)
        for i in range(1, 11):
            nums.append(weight*i)
        result.append(nums)

    for _ in range(250):
        nums = []
        amplitude = random.uniform(1.0, 5.0)      # amplitude of the sine wave
        frequency = random.uniform(0.1, 1.0)      # frequency (controls wave frequency)
        phase = random.uniform(0, 2 * math.pi)    # phase shift
        offset = random.uniform(-5, 5)            # vertical shift
        noise_scale = random.uniform(0, 2)

        for i in range(1, 11):
            noise = random.uniform(-noise_scale, noise_scale)
            val = amplitude * math.sin(frequency * i + phase) + offset + noise
            nums.append(round(val, 2))
        result.append(nums)

    for _ in range(250):
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
        
    for _ in range(250):
        nums = []
        base = random.uniform(1.05, 1.5)         # Growth/decay base
        scale = random.uniform(1, 3)             # Vertical stretch
        offset = random.uniform(-10, 10)
        noise_scale = random.uniform(0, 2)

        for i in range(1, 11):
            noise = random.uniform(-noise_scale, noise_scale)
            val = scale * (base ** i) + offset + noise
            nums.append(round(val, 2))

        result.append(nums)
        
    for _ in range(250):
        nums = []
        a = random.uniform(-0.2, 0.2)   # Cubic coefficient
        b = random.uniform(-1, 1)       # Quadratic
        c = random.uniform(-5, 5)       # Linear
        d = random.uniform(-10, 10)     # Constant
        noise_scale = random.uniform(0, 2)

        for i in range(1, 11):
            noise = random.uniform(-noise_scale, noise_scale)
            val = a*(i**3) + b*(i**2) + c*i + d + noise
            nums.append(round(val, 2))

        result.append(nums)

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