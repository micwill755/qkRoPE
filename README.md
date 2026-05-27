# qkRoPE Learning Notes

This is a small learning project for building intuition from self-attention to RoPE:

1. understand self-attention
2. build one context vector by hand
3. prove the rotation matrix
4. connect rotations to RoPE
5. walk through the pure-Python transformer code
6. extend RoPE to 2D RoPE and AS2DRoPE

There is no PyTorch and no NumPy. The point is readability, not speed.

## Recommended Order

Start here:

1. [attention.md](attention.md)

   Learn self-attention first. This guide walks through Q, K, V, attention scores, softmax weights, and how to build a context vector for one word. It ends by pointing out that attention alone does not include position.

```bash
python3 attention.py
```

2. [rotation_matrix_math_exercises.md](rotation_matrix_math_exercises.md)

   Work through the trigonometry and rotation matrix proof. This builds the math needed to understand why RoPE uses the matrix:

```text
[[cos θ, -sin θ],
 [sin θ,  cos θ]]
```

3. [code_walkthrough_notes.md](code_walkthrough_notes.md)

   Walk through the transformer/RoPE code in detail. This connects attention, Q/K dot products, embedding-dimension pairs, RoPE, 2D RoPE, and AS2DRoPE.

```bash
python3 transformer_walkthrough.py
```

Use [transformer_walkthrough.py](transformer_walkthrough.py) as the runnable companion to the code walkthrough.

## Mental Model

Transformers need position information because self-attention does not naturally know order or image location.

For text, a token has a 1D position:

```text
0, 1, 2, 3, ...
```

For images, a patch has a 2D position:

```text
row, column
```

## Absolute Position Embeddings

The classic learned approach stores a table:

```text
position 0 -> learned vector
position 1 -> learned vector
position 2 -> learned vector
```

Then it adds the position vector to the token vector:

```text
input = token_embedding + position_embedding
```

In real models this table is learned by gradient descent. In this walkthrough, we hand-create a tiny table so the operation is visible.

Absolute position embeddings are usually added before Q, K, and V are produced:

```text
input_with_position = token_embedding + position_embedding
Q = input_with_position @ W_query
K = input_with_position @ W_key
V = input_with_position @ W_value
```

So absolute position is mixed into the token representation first.

## Attention Scores

Attention starts by producing three views of the input:

```text
Q = what am I looking for?
K = how should I be found?
V = what information do I provide?
```

The dot product `QK^T` creates attention scores:

```text
scores = QK^T
```

Those scores answer:

```text
which tokens should this token look at?
```

Then softmax turns the scores into weights, and those weights are used to mix the value vectors:

```text
output = softmax(QK^T) V
```

That is why RoPE is applied to Q and K: QK is where the attention pattern is decided.

## RoPE

RoPE means Rotary Positional Embedding.

Instead of adding a position vector to the token, it rotates the query and key vectors inside attention:

```text
query = rotate(query, position)
key   = rotate(key, position)
```

This makes attention naturally sensitive to relative distance between positions.

The important detail:

```text
tokens choose the angle
embedding-dimension pairs get rotated
```

If Q/K dimension is `24`, each token has `12` two-dimensional pairs:

```text
(q0, q1), (q2, q3), ..., (q22, q23)
```

Each pair is a small 2D rotation plane. RoPE applies the rotation matrix to each pair.

The same frequencies are shared across all tokens, but each token multiplies them by its own position:

```text
token position m:
pair 0 -> mθ0
pair 1 -> mθ1
pair 2 -> mθ2
...
```

So each token does not get the same angles. It gets the same frequency set, scaled by its position.

This is why RoPE can carry position without adding a position vector.

It changes how Q and K compare:

```text
rotate(q_m, m) · rotate(k_n, n)
```

The comparison naturally contains the relative offset:

```text
m - n
```

One useful mental model:

```text
RoPE is many little unit-circle rotations happening side by side inside the embedding dimension.
```

## 2D RoPE

For image patches, position is not just `token_index`. It is:

```text
row, column
```

The demo splits the vector in half:

```text
first half  -> rotated by row
second half -> rotated by column
```

That gives attention separate awareness of vertical and horizontal offsets.

## AS2DRoPE

AS2DRoPE adds interpolation scaling for resolution changes.

If a model trained on a `14 x 14` patch grid but receives a `28 x 28` patch grid, raw positions would go beyond the training range:

```text
training: 0..13
current:  0..27
```

So the position is scaled:

```text
scaled_row = row * training_rows / current_rows
scaled_col = col * training_cols / current_cols
```

Example:

```text
row 20 in a 28-row grid -> 20 * 14 / 28 = 10
```

The high-resolution grid is squeezed back into the coordinate range the model learned.
