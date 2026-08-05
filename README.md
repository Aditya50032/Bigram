

This repository contains my notes, code, and exercises from **Andrej Karpathy's Makemore Lecture 1**.

## Topics Covered

- Bigram Language Model
- Character-level Language Modeling
- Probability Matrix
- Counting Character Pairs
- Laplace Smoothing
- Maximum Likelihood Estimation (MLE)
- Log Likelihood
- Negative Log Likelihood (NLL)
- Sampling New Names
- Basic PyTorch Operations

---

## Dataset

The dataset consists of thousands of names.

Example:

```
emma
olivia
ava
isabella
...
```

Each word is surrounded by start/end tokens (`.`).

Example:

```
emma

. -> e
e -> m
m -> m
m -> a
a -> .
```

---

## Bigram Model

A **Bigram** predicts the next character using only the previous character.

Example:

```
P(next_character | current_character)
```

We first count all character transitions.

Example Count Matrix:

| From | To | Count |
|------|----|-------|
| . | a | 4410 |
| a | b | 130 |
| a | . | 6640 |

---

## Probability Matrix

Counts are converted into probabilities.

```
P = Counts / Row Sum
```

Each row sums to **1**.

---

## Laplace Smoothing

Some transitions never appear in the dataset.

Without smoothing:

```
P = 0
log(0) = -∞
```

To avoid this:

```
P = N + 1
P /= P.sum(dim=1, keepdim=True)
```

Every probability becomes non-zero.

---

## Log Likelihood

Instead of multiplying many probabilities:

```
P1 × P2 × P3 × ...
```

Take logarithms:

```
log(P1) + log(P2) + log(P3)
```

Advantages:

- Prevents numerical underflow
- Easier optimization
- Faster computation

---

## Negative Log Likelihood (Loss)

Loss is defined as

```
Loss = -LogLikelihood / NumberOfExamples
```

Lower loss means better predictions.

Example:

```
log_likelihood = -38.7856

n = 16

Loss = 38.7856 / 16
     ≈ 2.42
```

---

## Sampling Names

Generate names character by character.

Example:

```
.

↓

m

↓

a

↓

x

↓

.
```

Stop generation when `.` is sampled.

---

## PyTorch Concepts Used

- torch.zeros()
- torch.tensor()
- torch.multinomial()
- torch.Generator()
- torch.log()
- Tensor indexing
- Broadcasting
- Normalization

---

## Common Mistakes I Encountered

### 1. log(0)

Problem

```
tensor(-inf)
```

Solution

```
Laplace Smoothing
```

---

### 2. Average Loss = inf

Cause

```
n = 0
```

Solution

```
n += 1
```

inside the inner loop.

---

### 3. Wrong Matrix Index

Wrong

```python
prob = P[ix, ix2]
```

Correct

```python
prob = P[ix1, ix2]
```

---

### 4. Forgot Normalization

Wrong

```python
P = N.float()
```

Correct

```python
P = (N + 1).float()
P /= P.sum(1, keepdim=True)
```

---

## Complexity

Building Count Matrix

```
Time : O(total characters)
Space : O(27 × 27)
```

Sampling

```
Time : O(length of generated word)
```

---

## Key Learnings

- Probability distributions
- Character-level language modeling
- Matrix normalization
- Log likelihood
- Maximum Likelihood Estimation
- Laplace smoothing
- Negative Log Likelihood
- Random sampling
- Basic PyTorch tensor operations

---

## Resources

- Andrej Karpathy - Neural Networks: Zero to Hero
- PyTorch Documentation
- Makemore GitHub Repository

---

## Author

**Your Name**

Learning Deep Learning one lecture at a time 🚀
