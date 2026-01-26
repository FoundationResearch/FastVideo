# Technical Specification: Context Forcing Distillation

## 1. Overview
[cite_start]Context Forcing is a novel distillation method proposed in HY-World 1.5 designed to distill a memory-augmented **Bidirectional Teacher** into a memory-aware **Autoregressive (AR) Student**[cite: 11, 334].

### The Core Problem: Distribution Mismatch
[cite_start]Standard distillation methods (like DMD) fail for memory-aware models due to a fundamental mismatch[cite: 345, 346]:
* **Teacher:** Bidirectional access (sees past and future).
* **Student:** Autoregressive access (causal, sees only past).
* **Result:** Even if both have memory mechanisms, their memory contexts ($C$) differ. [cite_start]This misalignment causes their conditional distributions $p(x|C)$ to diverge, making distribution matching impossible[cite: 349].

### The Solution: Context Forcing
We align the memory context of the Teacher to strictly match the Student's available context during the distillation process. [cite_start]This forces the Teacher to evaluate probabilities based *only* on the information available to the Student, enabling effective distribution matching[cite: 362, 368].

---

## 2. Mathematical Formulation

### 2.1 Distribution Matching Loss (DMD)
The goal is to align the student's distribution $p_{\theta}$ with the teacher's distribution. The gradient is approximated by the score difference:

$$
\nabla_{\theta}\mathcal{L}_{DMD}=\mathbb{E}_{k}(\nabla_{\theta}KL(p_{\theta}(x_{0:t})||p_{data}(x_{0:t})))
$$

[cite_start][cite: 341, 342]

### 2.2 Context Alignment Equation (The Core Mechanism)
To construct the Teacher model distribution $p_{data}$, we augment a standard bidirectional diffusion model with memory. Crucially, we structure its context by **masking** the target chunks ($x_{j:j+3}$) from the memory context:

$$
p_{data}(x_{j:j+3}|x_{0:j-1})=p_{\beta}(x_{j:j+3}|C_{j:j+3}-x_{j:j+3})
$$

Where:
* $x_{j:j+3}$: The sequence of chunks currently being generated (the target).
* $C_{j:j+3}$: The full context memory available for these chunks.
* $C_{j:j+3}-x_{j:j+3}$: The **Aligned Context ($C^{tea}$)**. [cite_start]We explicitly remove the target chunks from the Teacher's retrieval scope so it cannot "cheat" by looking at the ground truth of the current generation steps[cite: 364, 365, 367].

---

## 3. Algorithm: Context Forcing Training Loop

[cite_start]Based on **Algorithm 1** in the technical report[cite: 372].

### Inputs
* **Student Model ($N_{\theta}$):** AR Diffusion Transformer.
* **Teacher Models ($V_{\beta}^{fake}, V^{real}$):** Memory-Augmented Bidirectional Video Diffusion.
* **Hyperparameters:** Number of chunks $n=4$, Denoising steps $d$.

### Step-by-Step Execution Flow

1.  **Sampling & Initialization:**
    * Sample a start chunk index $j \sim \text{Uniform}(0, 1, ..., m)$.
    * Retrieve initial context $x_{0:j-1}$ from the dataset.

2.  **Student Self-Rollout (Autoregressive Generation):**
    * Loop for each chunk $i$ from $j$ to $j+n-1$:
        * Initialize latent $x_i \sim \mathcal{N}(0, I)$.
        * **Reconstitute Context ($C_i$):** Retrieve relevant memory chunks from the *student's own past history* $\{x_0, ..., x_{i-1}\}$.
        * **Multi-step Denoising:** Use $N_{\theta}$ to denoise $x_i$ using $s$ steps (sampled uniformly).
        * *Result:* A generated trajectory sequence $\hat{x}_{j:j+n-1}$.

3.  **Context Alignment (The "Forcing" Operation):**
    * Construct the Teacher's context $C^{tea}$.
    * **Logic:** $C^{tea} \leftarrow C_{j:j+n-1} - x_{j:j+n-1}$.
    * *Implementation Note:* Ensure the Teacher's memory retrieval mechanism is masked or restricted so it excludes the ground truth chunks $x_{j:j+n-1}$ corresponding to the current time steps.

4.  **Scoring & Update:**
    * Sample a diffusion timestep $k \sim [0, 1]$.
    * Add noise to the generated chunks: $\hat{x}_{noisy} \leftarrow \text{AddNoise}(\hat{x}_{j:j+n-1}, k)$.
    * **Compute Fake Score:** $S^{fake} \leftarrow V_{\beta}^{fake}(\hat{x}_{noisy}, C^{tea}, k)$.
    * **Compute Real Score:** $S^{real} \leftarrow V^{real}(\hat{x}_{noisy}, C^{tea}, k)$ (Note: Usually Real Score is computed on Ground Truth data $x_{GT}$ perturbed with noise, or using the generated data depending on the specific DMD variant. Eq 4 implies matching distributions. Algorithm 1 Line 14 uses $\hat{x}$ for Real Score computation, implying the gradient direction comes from the difference in scores at the student's data point). *Correction based on standard DMD: Usually we compare Score of Real Data vs Score of Fake Data, or Score Difference at the Fake Data point. Algorithm 1 implies calculating the score difference at the generated point $\hat{x}$.*

5.  **Loss Optimization:**
    * Update $\theta$ via distribution matching loss using $S^{real}$ and $S^{fake}$.

---

## 4. Architecture Diagram Description
[cite_start](Based on Figure 7 [cite: 360])

The architecture consists of two parallel branches connected by the Context Cache:

1.  **Left Branch (Student):**
    * **Process:** Memory-Augmented Self-Rollout.
    * **Mechanism:** It retrieves from the cache, updates the cache with its own predictions, and autoregressively produces 4 chunks.

2.  **Right Branch (Teachers):**
    * **Models:** Two copies of the Bidirectional Diffusion Model (one for Real Score, one for Fake Score).
    * **Input:** They receive the **same** Memory Context as the student (via the Context Forcing/Masking mechanism).
    * **Output:** Gradients (Scores) that guide the student.

## 5. Critical Implementation Details for Coding
1.  **Self-Rollout is Mandatory:** Do not use Teacher Forcing for the student loop. [cite_start]The student must consume its own predictions to expose error accumulation[cite: 363].
2.  **Masking Logic:** The `MemoryReconstitutor` for the Teacher must accept a `mask_indices` or `exclude_chunks` argument to implement the $C - x$ logic strictly.
3.  [cite_start]**Chunk-wise Processing:** The training happens on sequences of chunks (e.g., 4 chunks = 64 frames if 1 chunk = 16 frames), not single frames[cite: 204, 363].