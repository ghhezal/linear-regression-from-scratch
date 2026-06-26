import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("data/studytime_score.csv")


def cost_function(m, b, points):
    total_error = 0
    n = len(points)
    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(n)


def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].studytime
        y = points.iloc[i].score

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b


m = 0
b = 0
L = 0.001
epochs = 1000
costs = []

for i in range(epochs):
    if i % 1000 == 0:
        print(f"Epoch {i}: m={round(m, 3)}, b={round(b, 3)}, cost={round(cost_function(m, b, data), 3)}")
    m, b = gradient_descent(m, b, data, L)
    costs.append(cost_function(m, b, data))

print(f"\nFinal: m={round(m, 3)}, b={round(b, 3)}")
print(f"Final cost (MSE): {round(cost_function(m, b, data), 3)}")

# Plot 1: Regression line
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(data.studytime, data.score, color="black", label="Data")
plt.plot(list(range(0, 10)), [m * x + b for x in range(0, 10)], color="red", label="Regression line")
plt.xlabel("Study Time (hours)")
plt.ylabel("Score")
plt.title("Study Time vs Score — Linear Regression")
plt.legend()

# Plot 2: Loss curve
plt.subplot(1, 2, 2)
plt.plot(costs, color="blue")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Loss over Time")

plt.tight_layout()
plt.show()