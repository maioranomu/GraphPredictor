import numpy as np
import random

# Synthetic Graph Generators for Model Training
# Forward-thinking, modular and realistic time-series patterns

def generate_linear(length=10, slope=None, intercept=None, noise_level=0.0):
    x = np.arange(length)
    slope = slope if slope is not None else np.random.uniform(-2, 2)
    intercept = intercept if intercept is not None else np.random.uniform(-5, 5)
    noise = np.random.normal(scale=noise_level, size=length)
    return np.round(slope * x + intercept + noise, 2).tolist()

def generate_polynomial(length=10, coeffs=None, noise_level=0.0):
    x = np.arange(length)
    if coeffs is None:
        degree = np.random.randint(2, 5)
        coeffs = np.random.uniform(-1, 1, size=degree+1)
    poly = sum(c * x**i for i, c in enumerate(coeffs))
    noise = np.random.normal(scale=noise_level, size=length)
    return np.round(poly + noise, 2).tolist()

def generate_exponential(length=10, base=None, scale=None, noise_level=0.0):
    x = np.arange(length)
    base = base or np.random.uniform(1.01, 1.5)
    scale = scale or np.random.uniform(0.5, 2.0)
    noise = np.random.normal(scale=noise_level, size=length)
    return np.round(scale * np.power(base, x) + noise, 2).tolist()

def generate_logistic(length=10, L=None, k=None, x0=None, noise_level=0.0):
    x = np.arange(length)
    L = L or np.random.uniform(1, 10)
    k = k or np.random.uniform(0.1, 1.0)
    x0 = x0 or np.random.uniform(length/4, 3*length/4)
    curve = L / (1 + np.exp(-k * (x - x0)))
    noise = np.random.normal(scale=noise_level, size=length)
    return np.round(curve + noise, 2).tolist()

def generate_sinusoidal(length=10, amplitude=None, frequency=None, phase=None, offset=None, noise_level=0.0):
    x = np.arange(length)
    amplitude = amplitude or np.random.uniform(0.5, 5)
    frequency = frequency or np.random.uniform(0.1, 1.0)
    phase = phase or np.random.uniform(0, 2*np.pi)
    offset = offset or np.random.uniform(-2, 2)
    noise = np.random.normal(scale=noise_level, size=length)
    return np.round(amplitude * np.sin(2 * np.pi * frequency * x + phase) + offset + noise, 2).tolist()

def generate_random_walk(length=10, step_scale=1.0):
    steps = np.random.normal(scale=step_scale, size=length)
    walk = np.cumsum(steps)
    return np.round(walk, 2).tolist()

def generate_ar1(length=10, phi=None, noise_scale=1.0):
    phi = phi if phi is not None else np.random.uniform(-0.9, 0.9)
    y = np.zeros(length)
    noise = np.random.normal(scale=noise_scale, size=length)
    for t in range(1, length):
        y[t] = phi * y[t-1] + noise[t]
    return np.round(y, 2).tolist()

def generate_seasonal_trend(length=10, trend_slope=None, season_period=None, season_amp=None, noise_level=0.0):
    x = np.arange(length)
    trend = np.array(generate_linear(length, slope=trend_slope, intercept=0, noise_level=0.0))
    period = season_period or np.random.randint(2, length)
    amp = season_amp or np.random.uniform(0.5, 3)
    seasonal = amp * np.sin(2 * np.pi * x / period)
    noise = np.random.normal(scale=noise_level, size=length)
    return np.round(trend + seasonal + noise, 2).tolist()

# Mix and output with rewrite prompt
def generate_dataset(n_series=500, length=10):
    generators = [
        generate_linear, generate_polynomial, generate_exponential,
        generate_logistic, generate_sinusoidal, generate_random_walk,
        generate_ar1, generate_seasonal_trend
    ]
    dataset = []
    for _ in range(n_series):
        gen = random.choice(generators)
        kwargs = {}
        if 'noise_level' in gen.__code__.co_varnames:
            kwargs['noise_level'] = random.uniform(0, 1)
        series = gen(length=length, **kwargs)
        dataset.append(series)
    return dataset

if __name__ == "__main__":
    # Generate and shuffle data series for randomized export order
    data = generate_dataset(1000)
    random.shuffle(data)

    # Print each series
    for series in data:
        print(', '.join(map(str, series)))

    # Prepare export and shuffle rows
    export = [', '.join(map(str, row)) for row in data]
    random.shuffle(export)

    print("\nSample rows after shuffle:", export[:5])

    # Prompt user to rewrite file
    should_rewrite = input("REWRITE CURRENT DATA TO 'data.csv'? [Y/N]: ")
    if should_rewrite.strip().lower() == 'y':
        with open('data.csv', 'w') as f:
            f.write('n1,n2,n3,n4,n5,n6,n7,n8,n9,n10\n')
            f.write("\n".join(export))
        print("Data written to data.csv")
