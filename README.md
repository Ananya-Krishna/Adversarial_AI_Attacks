## Adversarial Techniques

This repository covers a breadth of adversarial techniques and defenses for ML attacks with a few select code implementations. This is a highly comprehensive resource meant for **educational purposes** but not an exhaustive list (created Summer 2025).

______

- **Gradient-Based Attacks (Expanded)**
  - *Fast Gradient Sign Method (FGSM)* — single‐step, first‐order attack using sign of the gradient.
  - *Projected Gradient Descent (PGD)* — iterative extension of FGSM that projects back into a norm ball.
  - *DeepFool* — approximately finds the minimal ℓ₂ perturbation required to cross the decision boundary by linearizing the classifier around the current point.  
  - *Carlini–Wagner (CW) Attack* — optimizes a protective loss (e.g., hinge) under various norms (ℓ₂, ℓ∞) to produce minimal, high‐confidence adversarial examples.
  - *Greedy Coordinate Gradients (GCG)* — a zeroth‐order or coordinate‐descent‐style approach that perturbs one input dimension at a time in the direction that maximally increases loss, repeating until misclassification.
  - *TurboAutoDan* (Auto‐Differentiable Adversarial Network) — uses a learned generator network (often a small Transformer or CNN) that, given a batch of clean inputs, outputs perturbations. Training alternates between:
    1. **Attack step**: update the generator’s parameters to maximize victim‐model loss.
    2. **Defense step**: optionally fine‐tune the victim model (via adversarial training).
  - *Momentum Iterative FGSM (MI‐FGSM)* — adds a momentum term to PGD to avoid local maxima in loss landscape.
  - *Jacobian‐based Saliency Map Attack (JSMA)* — computes a saliency map from the Jacobian ∂output/∂input to identify which pixels to perturb for a targeted misclassification.

- **Poisoning & Backdoor Attacks**
  - *Data Poisoning (Training‐Time)*  
    - *Label Flipping* — change labels of a small fraction of training samples so that at inference time, the model systematically misclassifies certain classes.  
    - *Feature Collision* — add malicious samples whose features are optimized to resemble target samples, causing the model to learn spurious associations.  
    - *Clean‐Label Poisoning* — craft training samples that look legitimately labeled but induce backdoor behavior when the victim model trains on them.
  - *Backdoor / Trojan Attacks*  
    - *Trigger Injection* — modify a small patch or trigger (e.g., a specific pattern of pixels) into a subset of training examples; at inference, the presence of that trigger forces a target label.  
    - *Invisible Triggers* — design perturbations in a subspace (e.g., singular vectors) that remain imperceptible but reliably activate a backdoor.

- **Black‐Box & Query‐Based Attacks**
  - *Zeroth‐Order Optimization (ZOO)* — estimate gradient via finite differences on output logits or loss, without API access to actual gradients.  
  - *Boundary Attack* — start from a large‐magnitude adversarial example and perform random walks toward the original image, staying just outside the decision boundary.  
  - *NES (Natural Evolution Strategies)* — sample random noise around the input, use the loss estimates to approximate gradients, then update adversarial perturbation.

- **Adaptive & Hybrid Attacks**
  - *TurboAutoDan* (described above)  
  - *Adaptive PGD* — combines PGD with dynamic step‐size or higher‐order gradient approximations for robustness‐targeted attacks.  
  - *Meta‐Learning Attacks* — use a small auxiliary dataset to meta‐train a generator that produces transferable perturbations against multiple models.

- **Domain‐Specific & Structured Attacks**
  - *Textual Attacks*  
    - *HotFlip* — use character‐level gradients to flip tokens (insert/delete/substitute) for misclassification.  
    - *TextFooler* — identify synonyms or paraphrases via word embeddings to replace words while preserving semantics.  
    - *Jailbreak Prompts* — craft input prompts that exploit model weaknesses in instruction‐following LLMs to bypass safety filters.
  - *Graph Attacks*  
    - *Nettack* — for node‐classification, modify a few edges or node features to change a target node’s classification.  
    - *MetaAttack* — use reinforcement learning or GANs to learn how to perturb graphs for maximal impact.
  - *Audio / Speech Attacks*  
    - *Psychoacoustic Hiding* — leverage human auditory masking to embed adversarial perturbations that are imperceptible but fool speech‐recognition models.  
    - *CommanderSong* — embed adversarial commands into benign audio samples so that ASR system transcribes attacker’s commands.

- **Defenses (Expanded)**
  - *Adversarial Training (Madry-style)* — train on strong PGD adversarial examples (often ℓ∞‐bounded) to improve robustness.  
  - *Feature Squeezing* — reduce input precision (e.g., bit‐depth reduction, JPEG compression) to remove adversarial noise.  
  - *Randomized Smoothing* — add random noise (usually Gaussian) at inference time and take a majority vote to certify robustness in an ℓ₂ ball.  
  - *Detection & Rejection* — train a separate “detector” network to identify and reject adversarial inputs (e.g., using statistical difference in activations).  
  - *Certified Defenses*  
    - *Interval Bound Propagation (IBP)* — propagate input intervals through the network to bound possible outputs, guaranteeing no misclassification within a norm ball.  
    - *Convex Relaxation* — relax ReLU networks into a convex outer polytope to compute certified margins.  

---
