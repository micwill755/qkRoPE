"""
Tiny pure-Python transformer walkthrough.

This file avoids PyTorch, NumPy, and training code on purpose. The goal is to
make the data flow visible:

1. absolute positional embeddings
2. Q/K/V and attention scores
3. RoPE as rotations inside embedding-dimension pairs
4. one self-attention transformer block
5. 2D RoPE for image patch grids
6. AS2DRoPE-style interpolation scaling for arbitrary image resolutions

Run:
    python3 transformer_walkthrough.py
"""

from __future__ import annotations

import math
import random
from typing import Iterable


Vector = list[float]
Matrix = list[Vector]


def zeros(n: int) -> Vector:
    return [0.0 for _ in range(n)]


def random_vector(n: int, scale: float = 0.02) -> Vector:
    return [random.uniform(-scale, scale) for _ in range(n)]


def random_matrix(rows: int, cols: int, scale: float = 0.02) -> Matrix:
    return [random_vector(cols, scale) for _ in range(rows)]


def add(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]


def sub(a: Vector, b: Vector) -> Vector:
    return [x - y for x, y in zip(a, b)]


def mul_scalar(a: Vector, s: float) -> Vector:
    return [x * s for x in a]


def dot(a: Vector, b: Vector) -> float:
    return sum(x * y for x, y in zip(a, b))


def matvec(x: Vector, weight: Matrix) -> Vector:
    """Multiply a vector by a weight matrix shaped [input_dim, output_dim]."""
    output_dim = len(weight[0])
    out = zeros(output_dim)
    for i, value in enumerate(x):
        for j in range(output_dim):
            out[j] += value * weight[i][j]
    return out


def matmul(a: Matrix, b: Matrix) -> Matrix:
    """Multiply matrices shaped [rows, shared] and [shared, cols]."""
    return [matvec(row, b) for row in a]


def transpose(matrix: Matrix) -> Matrix:
    return [list(col) for col in zip(*matrix)]


def softmax(values: Vector) -> Vector:
    max_value = max(values)
    exps = [math.exp(v - max_value) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def gelu(x: float) -> float:
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def print_matrix(name: str, matrix: Matrix, decimals: int = 3) -> None:
    print(f"\n{name}")
    for i, row in enumerate(matrix):
        formatted = ", ".join(f"{v: .{decimals}f}" for v in row)
        print(f"  {i}: [{formatted}]")


def print_vector(name: str, vector: Vector, decimals: int = 3) -> None:
    formatted = ", ".join(f"{v: .{decimals}f}" for v in vector)
    print(f"{name}: [{formatted}]")


class LayerNorm:
    """
    Per-token LayerNorm, kept here because VisionLLaMA reported it worked
    better than RMSNorm for their classification experiments.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = [1.0] * dim
        self.bias = [0.0] * dim

    def __call__(self, x: Matrix) -> Matrix:
        return self.forward(x)

    def forward(self, x: Matrix) -> Matrix:
        return [self._normalize_token(token) for token in x]

    def _normalize_token(self, token: Vector) -> Vector:
        mean = sum(token) / self.dim
        centered = [v - mean for v in token]
        variance = sum(v * v for v in centered) / self.dim
        scale = math.sqrt(variance + self.eps)
        return [(centered[i] / scale) * self.weight[i] + self.bias[i] for i in range(self.dim)]


def absolute_positional_embeddings(sequence: Matrix, pos_table: Matrix) -> Matrix:
    """
    Classic learned absolute position embeddings.

    Conceptually:
        output[token_index] = token_embedding + learned_position_vector[token_index]
    """
    return [add(token, pos_table[i]) for i, token in enumerate(sequence)]


def sinusoidal_position_vector(position: float, dim: int, base: float = 10000.0) -> Vector:
    """
    Fixed sinusoidal position embedding, included for contrast with learned
    absolute embeddings. This is not RoPE yet; it creates a vector to add.
    """
    values = zeros(dim)
    for i in range(0, dim, 2):
        inv_freq = 1.0 / (base ** (i / dim))
        angle = position * inv_freq
        values[i] = math.sin(angle)
        if i + 1 < dim:
            values[i + 1] = math.cos(angle)
    return values


def rotate_pair(x1: float, x2: float, angle: float) -> tuple[float, float]:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return x1 * cos - x2 * sin, x1 * sin + x2 * cos


def rope_1d(x: Vector, position: float, base: float = 10000.0) -> Vector:
    """
    Rotary Positional Embedding for a single token vector.

    RoPE rotates pairs of dimensions. Different pairs rotate at different
    frequencies, which gives the attention dot product a useful sense of
    relative position.
    """
    out = x[:]
    dim = len(x)
    for i in range(0, dim, 2):
        inv_freq = 1.0 / (base ** (i / dim))
        angle = position * inv_freq
        if i + 1 < dim:
            out[i], out[i + 1] = rotate_pair(x[i], x[i + 1], angle)
    return out


def rope_angles(position: float, dim: int, base: float = 10000.0) -> Vector:
    """Return the angle used for each 2D dimension pair at a token position."""
    return [position / (base ** (i / dim)) for i in range(0, dim, 2)]


def rope_2d(x: Vector, row: float, col: float, base: float = 10000.0) -> Vector:
    """
    2D RoPE for image patches.

    The first half of the vector is rotated by row position. The second half is
    rotated by column position. This lets attention compare vertical and
    horizontal offsets separately.
    """
    if len(x) % 4 != 0:
        raise ValueError("2D RoPE demo expects vector length divisible by 4.")

    half = len(x) // 2
    row_part = rope_1d(x[:half], row, base)
    col_part = rope_1d(x[half:], col, base)
    return row_part + col_part


def as2drope(
    x: Vector,
    row: int,
    col: int,
    current_grid: tuple[int, int],
    training_grid: tuple[int, int],
    base: float = 10000.0,
) -> Vector:
    """
    AS2DRoPE-style interpolation scaling.

    If the model trained on a 14x14 patch grid but currently sees a 28x28 grid,
    row 20 should not be treated as a totally new kind of position. We scale it
    back into the training coordinate system:

        scaled_row = row * training_rows / current_rows
        scaled_col = col * training_cols / current_cols
    """
    current_rows, current_cols = current_grid
    training_rows, training_cols = training_grid
    scaled_row = row * training_rows / current_rows
    scaled_col = col * training_cols / current_cols
    return rope_2d(x, scaled_row, scaled_col, base)


class Attention:
    """
    Single-head self-attention.

    This class owns the Q/K/V projections and the final output projection. RoPE
    is applied only to queries and keys, which is the usual place rotary
    position information enters attention.
    """

    def __init__(self, cfg: dict):
        self.emb_dim = cfg["emb_dim"]
        self.use_rope = cfg.get("use_rope", False)
        self.q_W = random_matrix(self.emb_dim, self.emb_dim, cfg["init_scale"])
        self.k_W = random_matrix(self.emb_dim, self.emb_dim, cfg["init_scale"])
        self.v_W = random_matrix(self.emb_dim, self.emb_dim, cfg["init_scale"])
        self.out_proj = random_matrix(self.emb_dim, self.emb_dim, cfg["init_scale"])

    def __call__(self, sequence: Matrix, positions: Iterable[float] | None = None) -> Matrix:
        return self.forward(sequence, positions)

    def forward(self, sequence: Matrix, positions: Iterable[float] | None = None) -> Matrix:
        queries = [matvec(x, self.q_W) for x in sequence]
        keys = [matvec(x, self.k_W) for x in sequence]
        values = [matvec(x, self.v_W) for x in sequence]

        if self.use_rope and positions is not None:
            positions = list(positions)
            queries = [rope_1d(q, positions[i]) for i, q in enumerate(queries)]
            keys = [rope_1d(k, positions[i]) for i, k in enumerate(keys)]

        output = []
        scale = 1.0 / math.sqrt(self.emb_dim)
        for q in queries:
            scores = [dot(q, k) * scale for k in keys]
            weights = softmax(scores)
            mixed = zeros(self.emb_dim)
            for weight, value in zip(weights, values):
                mixed = add(mixed, mul_scalar(value, weight))
            output.append(matvec(mixed, self.out_proj))
        return output


class FeedForward:
    """
    The per-token MLP used after attention.

    Attention mixes information across tokens. The feed-forward network then
    transforms each token independently.
    """

    def __init__(self, cfg: dict):
        self.fc1 = random_matrix(cfg["emb_dim"], cfg["hidden_dim"], cfg["init_scale"])
        self.fc2 = random_matrix(cfg["hidden_dim"], cfg["emb_dim"], cfg["init_scale"])

    def __call__(self, x: Vector) -> Vector:
        return self.forward(x)

    def forward(self, x: Vector) -> Vector:
        hidden = [gelu(v) for v in matvec(x, self.fc1)]
        return matvec(hidden, self.fc2)


class Block:
    """
    A small single-head transformer block.

    This block composes the two main sublayers:
        1. self-attention
        2. feed-forward network

    Each sublayer gets LayerNorm before it and a residual connection after it.
    """

    def __init__(self, cfg: dict):
        self.attn = Attention(cfg)
        self.feed_forward = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

    def __call__(self, sequence: Matrix, positions: Iterable[float] | None = None) -> Matrix:
        return self.forward(sequence, positions)

    def forward(self, x: Matrix, positions: Iterable[float] | None = None) -> Matrix:
        # Attention block with residual.
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, positions)
        x = [add(shortcut_token, attn_token) for shortcut_token, attn_token in zip(shortcut, x)]

        # Feed-forward block with residual.
        shortcut = x
        x = self.norm2(x)
        x = [self.feed_forward(token) for token in x]
        x = [add(shortcut_token, ff_token) for shortcut_token, ff_token in zip(shortcut, x)]

        return x


TinyTransformerBlock = Block


def demo_absolute_position_embeddings() -> None:
    print("\n=== 1. Absolute positional embeddings ===")
    sequence = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    pos_table = [
        [0.10, 0.00, 0.00, 0.00],
        [0.00, 0.20, 0.00, 0.00],
        [0.00, 0.00, 0.30, 0.00],
    ]
    print_matrix("token embeddings", sequence)
    print_matrix("learned absolute position table", pos_table)
    print_matrix("token + position", absolute_positional_embeddings(sequence, pos_table))


def demo_qkv_attention_scores() -> None:
    print("\n=== 2. Q, K, V and attention scores ===")
    inputs = [
        [0.43, 0.15, 0.89],  # the
        [0.55, 0.87, 0.66],  # cat
        [0.57, 0.85, 0.64],  # sat
        [0.22, 0.58, 0.33],  # on
        [0.77, 0.25, 0.10],  # the
        [0.05, 0.80, 0.55],  # mat
    ]
    w_query = [
        [0.10, 0.20],
        [0.30, 0.40],
        [0.50, 0.60],
    ]
    w_key = [
        [0.15, 0.25],
        [0.35, 0.45],
        [0.55, 0.65],
    ]
    w_value = [
        [0.20, 0.10],
        [0.40, 0.30],
        [0.60, 0.50],
    ]

    queries = matmul(inputs, w_query)
    keys = matmul(inputs, w_key)
    values = matmul(inputs, w_value)
    raw_scores = matmul(queries, transpose(keys))
    weights_for_first_token = softmax(raw_scores[0])

    print_matrix("inputs: 6 token embeddings, each dim 3", inputs)
    print_matrix("Q = inputs @ W_query: 6 query vectors, each dim 2", queries)
    print_matrix("K = inputs @ W_key: 6 key vectors, each dim 2", keys)
    print_matrix("V = inputs @ W_value: 6 value vectors, each dim 2", values)
    print_matrix("QK^T raw attention scores: each row asks 'who should I look at?'", raw_scores)
    print_vector("softmax scores for token 0", weights_for_first_token)
    mixed = zeros(len(values[0]))
    for weight, value in zip(weights_for_first_token, values):
        mixed = add(mixed, mul_scalar(value, weight))
    print_vector("token 0 output = weighted mixture of V rows", mixed)


def demo_rope_dimension_pairs() -> None:
    print("\n=== 3. RoPE rotates embedding-dimension pairs ===")
    dim = 24
    positions = list(range(6))
    print("With 6 tokens and Q/K dim 24:")
    print("  tokens choose the position multiplier")
    print("  dimensions are grouped into 12 two-dimensional rotation planes")
    print("  each pair uses a different frequency")

    for position in positions:
        angles = rope_angles(position, dim)
        first = ", ".join(f"{angle:.6f}" for angle in angles[:4])
        last = ", ".join(f"{angle:.6f}" for angle in angles[-3:])
        print(f"token position {position}: first angles [{first}, ...], last angles [{last}]")

    token_query = [1.0, 0.0] * 12
    position = 3
    rotated = rope_1d(token_query, position)
    print_vector("one token query before RoPE", token_query[:8])
    print_vector("same query after RoPE at position 3", rotated[:8])
    print("Notice: token position changes the angles; dimension pairs are what get rotated.")


def demo_rope_relative_scores() -> None:
    print("\n=== 4. RoPE makes QK scores relative-position aware ===")
    q = [1.0, 0.0]
    k = [1.0, 0.0]
    print("Use one 2D pair so the rotation is easy to see.")
    for q_pos, k_pos in [(0, 0), (0, 1), (0, 4), (2, 5)]:
        q_rot = rope_1d(q, q_pos, base=10000.0)
        k_rot = rope_1d(k, k_pos, base=10000.0)
        score = dot(q_rot, k_rot)
        print(f"q position {q_pos}, k position {k_pos}: dot={score:.3f}, offset={q_pos - k_pos}")
    print("Q and K are rotated by absolute positions, but their dot product exposes the offset.")


def demo_rope() -> None:
    print("\n=== 5. 1D RoPE on one token vector ===")
    token = [1.0, 0.0, 1.0, 0.0]
    for position in [0, 1, 2, 8]:
        print(f"position {position}: {rope_1d(token, position)}")


def demo_2d_rope() -> None:
    print("\n=== 6. 2D RoPE ===")
    patch = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    locations = [(0, 0), (0, 3), (2, 0), (2, 3)]
    for row, col in locations:
        rounded = [round(v, 3) for v in rope_2d(patch, row, col)]
        print(f"row={row}, col={col}: {rounded}")


def demo_as2drope() -> None:
    print("\n=== 7. AS2DRoPE interpolation scaling ===")
    patch = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    training_grid = (14, 14)
    current_grid = (28, 28)
    row, col = 20, 10
    scaled_row = row * training_grid[0] / current_grid[0]
    scaled_col = col * training_grid[1] / current_grid[1]
    print(f"training grid: {training_grid}")
    print(f"current grid:  {current_grid}")
    print(f"raw location:  row={row}, col={col}")
    print(f"scaled loc:    row={scaled_row}, col={scaled_col}")
    rounded = [round(v, 3) for v in as2drope(patch, row, col, current_grid, training_grid)]
    print(f"AS2DRoPE vector: {rounded}")


def demo_transformer_block() -> None:
    print("\n=== 8. Tiny transformer block with optional RoPE ===")
    random.seed(7)
    cfg = {
        "emb_dim": 4,
        "hidden_dim": 8,
        "init_scale": 0.2,
        "use_rope": False,
    }
    rope_cfg = {
        "emb_dim": 4,
        "hidden_dim": 8,
        "init_scale": 0.2,
        "use_rope": True,
    }
    sequence = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    block = Block(cfg)
    rope_block = Block(rope_cfg)
    print_matrix("input", sequence)
    print_matrix("output without RoPE", block(sequence))
    print_matrix("output with 1D RoPE in attention", rope_block(sequence, positions=[0, 1, 2]))


def main() -> None:
    demo_absolute_position_embeddings()
    demo_qkv_attention_scores()
    demo_rope_dimension_pairs()
    demo_rope_relative_scores()
    demo_rope()
    demo_2d_rope()
    demo_as2drope()
    demo_transformer_block()


if __name__ == "__main__":
    main()
