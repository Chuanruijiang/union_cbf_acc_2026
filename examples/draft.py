import numpy as np


def main():
    pi = np.pi
    x = pi/9
    print("compare sin(x) with Taylor expansion")
    print(x - x**3/(3*2))
    print(np.sin(x))
    print("compare cos(x) with Taylor expansion")
    print(1 - x**2/2)
    print(np.cos(x))


if __name__ == "__main__":
    main()
