# Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion

**Authors:** Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, Eli Shechtman **Institution:** Adobe Research, The University of Texas at Austin 

## Abstract

We introduce **Self Forcing**, a novel training paradigm for autoregressive video diffusion models. It addresses the longstanding issue of **exposure bias**, where models trained on ground-truth context must generate sequences conditioned on their own imperfect outputs during inference. Unlike prior methods that denoise future frames based on ground-truth context frames, Self Forcing conditions each frame's generation on previously self-generated outputs by performing autoregressive rollout with key-value (KV) caching during training. This strategy enables supervision through a holistic loss at the video level that directly evaluates the quality of the entire generated sequence, rather than relying solely on traditional frame-wise objectives.

To ensure training efficiency, we employ a few-step diffusion model along with a stochastic gradient truncation strategy, effectively balancing computational cost and performance. We further introduce a **rolling KV cache mechanism** that enables efficient autoregressive video extrapolation. Extensive experiments demonstrate that our approach achieves real-time streaming video generation with sub-second latency on a single GPU, while matching or even surpassing the generation quality of significantly slower and non-causal diffusion models.

---

## 1. Introduction

Recent years have witnessed tremendous progress in video synthesis, typically achieved with diffusion transformers (DiT) that denoise all frames simultaneously using bidirectional attention. However, this design requires generating the entire video at once, limiting applicability to real-time streaming where future information is unknown.

In contrast, **autoregressive (AR) models** generate videos sequentially, aligning with the causal structure of temporal media. This approach reduces latency and enables applications like interactive content creation and game simulation. However, AR models often struggle to match the visual fidelity of state-of-the-art video diffusion models due to reliance on lossy vector quantization.

To combine these approaches, techniques like **Teacher Forcing (TF)** and **Diffusion Forcing (DF)** have emerged.

* 
**Teacher Forcing:** Trains the model to predict the next token conditioned on ground-truth tokens (next-frame prediction).


* 
**Diffusion Forcing:** Trains on videos with noise levels independently sampled for each frame, denoising based on noisy context frames.



However, both suffer from **exposure bias**: the model is trained on ground-truth (or specific noise distribution) context but must rely on its own imperfect predictions at inference, leading to error accumulation.

We propose **Self Forcing (SF)**. Inspired by RNN-era techniques, we bridge the train-test gap by explicitly unrolling autoregressive generation during training. Each frame is generated conditioned on previously **self-generated frames**. This enables supervision with holistic distribution-matching losses (e.g., DMD, SiD, GAN) applied to complete sequences.

**Key Contributions:**

1. **Self Forcing Algorithm:** Addresses exposure bias by training on self-generated context. Efficiently implemented in post-training using few-step diffusion and gradient truncation.


2. 
**Rolling KV Cache:** Enhances efficiency for video extrapolation.


3. 
**Performance:** Enables real-time generation at 17 FPS with sub-second latency on a single H100 GPU, matching or surpassing slower bidirectional models.



---

## 2. Related Work

* **GANs for Video Generation:** Early approaches avoided exposure bias by ensuring the generator followed the same process during training and inference. Our work draws inspiration from this by optimizing the alignment between the generator's output distribution and the target distribution.


* 
**Autoregressive/Diffusion Models:** Modern models shifted to Diffusion (bidirectional attention) or Autoregressive (next-token prediction).


* **Hybrid Models:** Recent works integrate AR and diffusion. However, they often rely on long iterative prediction chains leading to error accumulation. We address this by training the model to correct its own mistakes.


* **CausVid:** Closely related to our work but suffers from a flaw where training outputs (via Diffusion Forcing) do not match the inference-time distribution. We propose a solution that matches the true model distribution.



---

## 3. Self Forcing: Bridging Train-Test Gap via Holistic Post-Training

### 3.1 Preliminaries: Autoregressive Video Diffusion Models

An autoregressive video diffusion model factorizes the joint distribution into a product of conditionals:


Each conditional  is modeled using a diffusion process.

Most existing models use **Teacher Forcing (TF)** or **Diffusion Forcing (DF)**.

* 
**TF:** Denoises frame  conditioned on ground truth .


* 
**DF:** Denoises frame  conditioned on noisy context.
Both use frame-wise MSE loss. We focus on transformer-based architectures (DiT) with causal attention.



### 3.2 Autoregressive Diffusion Post-Training via Self-Rollout

The core idea is to generate videos through **autoregressive self-rollout** during training, mirroring inference.

* 
**Process:** Sample a batch of videos where each frame is generated by iterative denoising conditioned on *self-generated* outputs (clean past context and noisy current frame).


* 
**KV Caching:** Unlike previous models, we employ KV caching *during training* to enable efficiency.


* 
**Few-Step Approximation:** To avoid prohibitive costs, we use a few-step diffusion model  (e.g., 4 steps).


* 
**Gradient Truncation:** We limit backpropagation to only the final denoising step of each frame and detach gradients of previous frames to manage memory.


* 
**Stochastic Sampling:** Instead of always using  steps, we randomly sample a step  to ensure all intermediate steps receive supervision.



**Algorithm 1: Self Forcing Training** 

```python
Require: Denoise timesteps {t_1, ..., t_T}
Require: Number of video frames N
Require: AR diffusion model G_theta

1: loop
2:    Initialize model output X <- []
3:    Initialize KV cache KV <- []
4:    Sample s ~ Uniform(1, 2, ..., T)
5:    
6:    for i = 1, ..., N do
7:       Initialize x_{t_T}^i ~ N(0, I)
8:       for j = T, ..., s do
9:          if j == s then
10:             Enable gradient computation
11:             x_hat_0^i <- G_theta(x_{t_j}^i; t_j, KV)
12:             X.append(x_hat_0^i)
13:             Disable gradient computation
14:             Cache kv^i <- G_theta_KV(x_hat_0^i; 0, KV)
15:             KV.append(kv^i)
16:         else
17:             Disable gradient computation
18:             x_hat_0^i <- G_theta(x_{t_j}^i; t_j, KV)
19:             Sample epsilon ~ N(0, I)
20:             x_{t_{j-1}}^i <- Psi(x_hat_0^i, epsilon, t_{j-1})
21:         end if
22:      end for
23:   end for
24:   Update theta via distribution matching loss (on X)
25: end loop

```

### 3.3 Holistic Distribution Matching Loss

We apply video-level losses to align generated video distribution  with real data . We inject noise to both and match  and . We consider three objectives:

1. 
**DMD (Distribution Matching Distillation):** Minimizes reverse KL divergence using score difference.




2. 
**SiD (Score Identity Distillation):** Minimizes Fisher divergence.


3. 
**GANs:** Minimizes Jensen-Shannon divergence using a discriminator.



Crucially, we match the **holistic distribution** of the entire sequence , unlike TF/DF which perform frame-wise matching.

### 3.4 Long Video Generation with Rolling KV Cache

Standard sliding-window inference is inefficient.

* 
**Bidirectional models:** Cannot use KV cache ().


* 
**Causal models (Standard):** Require recomputing KV cache for overlaps ().



We propose a **Rolling KV Cache**:

* Maintain a fixed-size cache of size .
* When full, remove the oldest entry before adding the new one.
* Achieves  complexity.



**Algorithm 2: AR Inference with Rolling KV Cache** 

* Uses a queue-like structure for KV cache (pop(0) when full).
* **Fixing Distribution Mismatch:** Naive rolling cache causes artifacts because the model is trained seeing the first frame (image latent) which disappears from the cache window.
* 
**Solution:** During training, restrict the attention window so the model *cannot* attend to the first chunk when denoising the final chunk, simulating the rolling cache scenario.



---

## 4. Experiments

**Implementation:**

* Base Model: Wan2.1-T2V-1.3B (16 FPS, 832x480).


* Training: Finetuned with causal masking, then Self Forcing using R3GAN, DMD, or SiD objectives.


* Hardware: Single NVIDIA H100 GPU for speed tests.



**Main Results:**

**Table 1: Comparison with Baselines** 

| Model | #Params | Resolution | Throughput (FPS) | Latency (s) | Total Score | Quality Score | Semantic Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Diffusion Models** |  |  |  |  |  |  |  |
| LTX-Video | 1.9B | 768x512 | 8.98 | 13.5 | 80.00 | 82.30 | 70.79 |
| Wan2.1 | 1.3B | 832x480 | 0.78 | 103 | 84.26 | 85.30 | 80.09 |
| **Chunk-wise AR** |  |  |  |  |  |  |  |
| SkyReels-V2 | 1.3B | 960x540 | 0.49 | 112 | 84.70 | 82.67 | 74.53 |
| MAGI-1 | 4.5B | 832x480 | 0.19 | 282 | 79.18 | 82.04 | 67.74 |
| CausVid | 1.3B | 832x480 | 17.0 | 0.69 | 81.20 | 84.05 | 69.80 |
| **Self Forcing (Ours)** | **1.3B** | **832x480** | **17.0** | **0.69** | **84.31** | **85.07** | **81.28** |
| **Frame-wise AR** |  |  |  |  |  |  |  |
| **Self Forcing (Ours)** | **1.3B** | **832x480** | **8.9** | **0.45** | **80.30** | **84.26** | **85.25** |

* 
**Performance:** Self Forcing (chunk-wise) achieves the highest VBench scores while delivering real-time throughput (17.0 FPS).


* 
**User Study:** Self Forcing is consistently preferred over baselines, including the base model Wan2.1.



**Ablation Studies (Table 2):** 

* Self Forcing outperforms Teacher Forcing (TF) and Diffusion Forcing (DF) across all metrics (DMD, SiD, GAN).
* TF and DF show degradation when switching to frame-wise generation due to error accumulation; Self Forcing remains robust.



**Training Efficiency:**

* Self Forcing matches per-iteration time of TF/DF (due to efficient FlashAttention vs. masked attention).


* Self Forcing achieves superior quality for the same wall-clock training time.



---

## 5. Discussion

* **Parallelization Limit:** Parallel training creates a misalignment between training and inference distributions. We advocate for parallel pre-training followed by sequential post-training.


* **AR/Diffusion/GAN Integration:** We show these are complementary. The GAN principle (matching generator distribution) can train AR-Diffusion models.


* **Limitations:** Quality degrades for videos substantially longer than training context. Gradient truncation may limit learning long-range dependencies.



---

## Appendices

### A. Implementation Details

**Training Hyperparameters (Table 3):** 

| Hyperparameters | DMD | SiD | GAN |
| --- | --- | --- | --- |
| **Real score network** | Wan2.1-14B | Wan2.1-1.3B | N/A |
| **Batch size** | 64 | 64 | 768 |
| **Optimizer** | AdamW | Adam | AdamW |
| **Gen LR** | 2e-6 | 2e-6 | 2e-6 |
| **Gen/Critic Ratio** | 5 | 5 | 1 |

We use the Wan2.1 flow matching framework. The forward process is .

### B. Rolling KV Cache Ablation

We compare our local attention training against a naive baseline for video extrapolation.

* **Naive:** Model always attends to the first chunk. Fails when the first chunk is evicted from the cache.
* **Ours:** Restricts attention window during training. Mitigates artifacts and maintains high throughput (16.1 FPS).



### C. VBench Scores

Self Forcing generally outperforms other models in semantic alignment (scene, object class, human action). Frame-wise AR has higher dynamic degree but lower temporal consistency than chunk-wise AR.