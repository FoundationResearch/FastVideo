### 6. Model Post-Training II: Context Forcing Distillation

This post-training stage introduces **Context Forcing**, a distillation method designed to enable real-time performance while preserving the memory capabilities developed in previous training stages. This approach specifically addresses the challenge of distribution mismatch between bidirectional teacher models and autoregressive student models, which has historically prevented effective distillation in memory-aware generative systems.

#### 6.1 The Distribution Matching Challenge

Autoregressive world models typically suffer from error accumulation during long video generation and require slow denoising processes. While recent methods have attempted to address this by distilling a bidirectional diffusion model into a few-step autoregressive student, these techniques often fail for memory-aware models.

**The Standard Approach:**
Standard techniques force the student's distribution  to align with the teacher's using a distribution matching loss:

**The Mismatch Problem:**
These methods fail because of fundamental distribution differences between the teacher and the student:

* 
**The Teacher:** A bidirectional model that accesses full context (both past and future frames).


* 
**The Student:** An autoregressive model that can only access past context due to causal generation requirements.



This mismatch is critical for memory-aware models where the student relies on sophisticated memory mechanisms. Even if the teacher is augmented with memory, the difference in context access causes the conditional distributions  to misalign, causing distribution matching to fail.

#### 6.2 Context Forcing

To solve the mismatch, **Context Forcing** aligns the memory context between the teacher and the student during distillation (See Figure 7 in source).

**The Process:**

1. 
**Student Model:** Performs a self-rollout of 4 chunks conditioned on the memory context .


2. **Teacher Model:** A standard bidirectional diffusion model () is augmented with memory. To align with the student, the teacher's context is structured by masking the current target frames () from the student's memory context.



The teacher's distribution is formulated as:

Where  denotes all context memory chunks corresponding to the student's self-rollout .

**Benefits:**

* 
**Alignment:** By aligning the memory context, the distributions represented by the teacher become as close as possible to the student model, enabling effective distribution matching.


* 
**Performance:** This preserves long-term consistency in real-time generation (using only 4 denoising steps) and mitigates error accumulation over long sequences.



---

#### Algorithm 1: Context Forcing Training

**Require:**

* Number of denoising timesteps  and chunks  


* Dataset  (encoded by 3D VAE) 


* AR diffusion model  


* Bidirectional diffusion model  and  



```python
1:  loop
2:      Progressively increase maximum chunk length m
3:      Sample chunk length j ~ Uniform(0, 1, ..., m)
4:      Sample context x_{0:j-1} ~ D
5:      
6:      # Student Self-Rollout
7:      for i = j, ..., j + n - 1 do
8:          Initialize x_i^{init} ~ N(0, I)
9:          Reconstitute context memory C_i subset {x_0, ..., x_{i-1}}
10:         Sample s ~ Uniform(1, 2, ..., d)
11:         Self-rollout x_i using N_{\theta} with C_i and s denoising steps
12:     end for
13:     
14:     # Context Alignment and Scoring
15:     Align context memory C^{tea} <- C_{j:j+n-1} - x_{j:j+n-1}
16:     Sample diffusion timestep k ~ [0, 1]
17:     x_hat_{j:j+n-1} <- AddNoise(x_{j:j+n-1}, k)
18:     
19:     Compute fake score S^{fake} <- V_{\beta}^{fake}(x_hat_{j:j+n-1}, C^{tea}, k)
20:     Compute real score S^{real} <- V^{real}(x_hat_{j:j+n-1}, C^{tea}, k)
21:     
22:     Update theta via distribution matching loss
23:     Update beta via flow matching loss as in [8]
24: end loop

```