"""
Pure-Python self-attention walkthrough.

This file intentionally avoids PyTorch/NumPy so every operation is visible.
It follows one sentence:

    the cat sat on the mat

The goal is to show:

1. how inputs produce Q, K, V
2. how one token's query compares with all keys
3. how softmax turns scores into weights
4. how weights mix value vectors into a context vector
5. how the same process is done for every token at once

Run:
    python3 attention.py
"""

from __future__ import annotations

import math


Vector = list[float]
Matrix = list[Vector]


WORDS = ["the", "cat", "sat", "on", "the", "mat"]

INPUTS: Matrix = [
    [0.43, 0.15, 0.89],  # the  (x^1)
    [0.55, 0.87, 0.66],  # cat  (x^2)
    [0.57, 0.85, 0.64],  # sat  (x^3)
    [0.22, 0.58, 0.33],  # on   (x^4)
    [0.77, 0.25, 0.10],  # the  (x^5)
    [0.05, 0.80, 0.55],  # mat  (x^6)
]

W_QUERY: Matrix = [
    [0.10, 0.20],
    [0.30, 0.40],
    [0.50, 0.60],
]

W_KEY: Matrix = [
    [0.15, 0.25],
    [0.35, 0.45],
    [0.55, 0.65],
]

W_VALUE: Matrix = [
    [0.20, 0.10],
    [0.40, 0.30],
    [0.60, 0.50],
]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def matvec(x: Vector, weight: Matrix) -> Vector:
    output_dim = len(weight[0])
    out = [0.0] * output_dim
    for i, value in enumerate(x):
        for j in range(output_dim):
            out[j] += value * weight[i][j]
    return out


def matmul(a: Matrix, b: Matrix) -> Matrix:
    return [matvec(row, b) for row in a]


def transpose(matrix: Matrix) -> Matrix:
    return [list(col) for col in zip(*matrix)]


def softmax(values: Vector) -> Vector:
    max_value = max(values)
    exps = [math.exp(v - max_value) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def weighted_sum(weights: Vector, values: Matrix) -> Vector:
    out = [0.0] * len(values[0])
    for weight, value in zip(weights, values):
        for i in range(len(out)):
            out[i] += weight * value[i]
    return out


def print_vector(name: str, vector: Vector, decimals: int = 4) -> None:
    values = ", ".join(f"{v:.{decimals}f}" for v in vector)
    print(f"{name} = [{values}]")


def print_matrix(name: str, matrix: Matrix, labels: list[str] | None = None, decimals: int = 4) -> None:
    print(f"\n{name} = [")
    for i, row in enumerate(matrix):
        values = ", ".join(f"{v:.{decimals}f}" for v in row)
        suffix = f"  # {labels[i]}" if labels else ""
        print(f"  [{values}],{suffix}")
    print("]")


def explain_shapes() -> None:
    print("\n=== Shapes ===")
    print("inputs: 6 tokens x 3 embedding dimensions")
    print("W_query: 3 input dims x 2 output dims")
    print("W_key:   3 input dims x 2 output dims")
    print("W_value: 3 input dims x 2 output dims")
    print("Q, K, V: 6 tokens x 2 projected dimensions")


def calculate_single_context_vector() -> None:
    print("\n=== Context vector for one word: cat ===")
    cat_index = 1
    x_2 = INPUTS[cat_index]

    query_2 = matvec(x_2, W_QUERY)
    keys = matmul(INPUTS, W_KEY)
    values = matmul(INPUTS, W_VALUE)

    print_vector("x^2", x_2)
    print_vector("query_2 = x^2 @ W_query", query_2)
    print_matrix("keys = inputs @ W_key", keys, WORDS)
    print_matrix("values = inputs @ W_value", values, WORDS)

    attention_scores = [dot(query_2, key) for key in keys]
    attention_weights = softmax(attention_scores)
    context_vector_2 = weighted_sum(attention_weights, values)

    print_vector("attention_scores = query_2 @ keys.T", attention_scores)
    print_vector("attention_weights = softmax(attention_scores)", attention_weights)
    print(f"sum(attention_weights) = {sum(attention_weights):.4f}")
    print_vector("context_vector_2 = attention_weights @ values", context_vector_2)

    best_index = max(range(len(attention_scores)), key=lambda i: attention_scores[i])
    print(f'Highest raw score for "cat": {WORDS[best_index]}')


def calculate_all_context_vectors() -> None:
    print("\n=== Context vectors for all words ===")
    queries = matmul(INPUTS, W_QUERY)
    keys = matmul(INPUTS, W_KEY)
    values = matmul(INPUTS, W_VALUE)

    attention_scores = matmul(queries, transpose(keys))
    attention_weights = [softmax(row) for row in attention_scores]
    context_vectors = matmul(attention_weights, values)

    print_matrix("queries = inputs @ W_query", queries, WORDS)
    print_matrix("keys = inputs @ W_key", keys, WORDS)
    print_matrix("values = inputs @ W_value", values, WORDS)
    print_matrix("attention_scores = queries @ keys.T", attention_scores, WORDS)
    print_matrix("attention_weights = softmax(each score row)", attention_weights, WORDS)
    print_matrix("context_vectors = attention_weights @ values", context_vectors, WORDS)

    print("\nEach row in context_vectors is one updated token representation.")
    print("inputs shape:          6 x 3")
    print("context_vectors shape: 6 x 2")


def main() -> None:
    explain_shapes()
    calculate_single_context_vector()
    calculate_all_context_vectors()


if __name__ == "__main__":
    main()
