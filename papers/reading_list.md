# Reading List

Organized chronologically by when you should read them in the learning path. Papers marked with * are essential — read these first.

---

## Phase 0: Mathematics Foundations

### Linear Algebra & Geometry
- *Strang, G. "Introduction to Linear Algebra" — Chapters 1-7 (the standard reference)
- *3Blue1Brown "Essence of Linear Algebra" — YouTube series (geometric intuition)

### Calculus & Optimization
- *Boyd & Vandenberghe "Convex Optimization" — Chapters 1-5 (free online)
- Nocedal & Wright "Numerical Optimization" — Chapters 1-7

### Probability & Statistics
- *Jaynes "Probability Theory: The Logic of Science" — Chapters 1-4 (Bayesian foundations)
- Bishop "Pattern Recognition and Machine Learning" — Chapters 1-2

### Information Theory
- *Cover & Thomas "Elements of Information Theory" — Chapters 1-4
- *Shannon (1948) "A Mathematical Theory of Communication" — the original paper
- *Tishby et al. (2000) "The Information Bottleneck Method"

---

## Phase 1: Neural Network Foundations

### The Basics
- *Rosenblatt (1958) "The Perceptron: A Probabilistic Model"
- *Rumelhart, Hinton, Williams (1986) "Learning Representations by Back-propagating Errors" — backprop paper
- *Hornik (1991) "Approximation Capabilities of Multilayer Feedforward Networks" — universal approximation
- Cybenko (1989) "Approximation by Superpositions of a Sigmoidal Function"

### Initialization & Training
- *Glorot & Bengio (2010) "Understanding the Difficulty of Training Deep Feedforward Neural Networks" — Xavier init
- *He et al. (2015) "Delving Deep into Rectifiers" — He initialization, PReLU
- *Ioffe & Szegedy (2015) "Batch Normalization: Accelerating Deep Network Training"
- Srivastava et al. (2014) "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"

### Activation Functions
- *Nair & Hinton (2010) "Rectified Linear Units Improve Restricted Boltzmann Machines" — ReLU
- Hendrycks & Gimpel (2016) "Gaussian Error Linear Units (GELUs)"
- Ramachandran et al. (2017) "Searching for Activation Functions" — Swish

---

## Phase 2: Classical Architectures

### Convolutional Neural Networks
- *LeCun et al. (1998) "Gradient-Based Learning Applied to Document Recognition" — LeNet
- *Krizhevsky et al. (2012) "ImageNet Classification with Deep CNNs" — AlexNet
- *Simonyan & Zisserman (2014) "Very Deep Convolutional Networks" — VGGNet
- *He et al. (2016) "Deep Residual Learning for Image Recognition" — ResNet
- Szegedy et al. (2015) "Going Deeper with Convolutions" — Inception

### Recurrent Neural Networks
- *Hochreiter & Schmidhuber (1997) "Long Short-Term Memory" — LSTM
- Cho et al. (2014) "Learning Phrase Representations using RNN Encoder-Decoder" — GRU
- *Pascanu et al. (2013) "On the Difficulty of Training Recurrent Neural Networks" — vanishing gradients

### Autoencoders
- Hinton & Salakhutdinov (2006) "Reducing the Dimensionality of Data with Neural Networks"
- Vincent et al. (2008) "Extracting and Composing Robust Features with Denoising Autoencoders"

---

## Phase 3: Representation Learning

### Word Embeddings
- *Mikolov et al. (2013) "Efficient Estimation of Word Representations in Vector Space" — Word2Vec
- *Mikolov et al. (2013) "Distributed Representations of Words and Phrases and their Compositionality" — negative sampling
- Pennington et al. (2014) "GloVe: Global Vectors for Word Representation"

### Representation Theory
- *Bengio et al. (2013) "Representation Learning: A Review and New Perspectives"
- Tishby & Zaslavsky (2015) "Deep Learning and the Information Bottleneck Principle"
- Shwartz-Ziv & Tishby (2017) "Opening the Black Box of Deep Neural Networks via Information"

---

## Phase 4: Modern Architectures (Transformers)

### Attention
- *Bahdanau et al. (2015) "Neural Machine Translation by Jointly Learning to Align and Translate" — attention mechanism
- Luong et al. (2015) "Effective Approaches to Attention-based Neural Machine Translation"

### The Transformer
- *Vaswani et al. (2017) "Attention Is All You Need" — THE transformer paper
- *Devlin et al. (2019) "BERT: Pre-training of Deep Bidirectional Transformers"
- *Radford et al. (2018) "Improving Language Understanding by Generative Pre-Training" — GPT
- *Radford et al. (2019) "Language Models are Unsupervised Multitask Learners" — GPT-2
- Brown et al. (2020) "Language Models are Few-Shot Learners" — GPT-3

### Vision Transformers
- *Dosovitskiy et al. (2021) "An Image is Worth 16x16 Words" — ViT
- Liu et al. (2021) "Swin Transformer: Hierarchical Vision Transformer"

### Efficient Transformers
- Kitaev et al. (2020) "Reformer: The Efficient Transformer"
- Katharopoulos et al. (2020) "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
- Choromanski et al. (2021) "Rethinking Attention with Performers"

---

## Phase 5: Learning Dynamics

### Optimization
- *Kingma & Ba (2015) "Adam: A Method for Stochastic Optimization"
- *Loshchilov & Hutter (2019) "Decoupled Weight Decay Regularization" — AdamW
- Smith (2017) "Cyclical Learning Rates for Training Neural Networks"

### Loss Landscapes & Generalization
- *Li et al. (2018) "Visualizing the Loss Landscape of Neural Nets"
- *Keskar et al. (2017) "On Large-Batch Training: Generalization Gap and Sharp Minima"
- Belkin et al. (2019) "Reconciling Modern ML Practice and the Bias-Variance Trade-off" — double descent
- *Nakkiran et al. (2021) "Deep Double Descent"

### Generalization Theory
- Zhang et al. (2017) "Understanding Deep Learning Requires Rethinking Generalization"
- Jacot et al. (2018) "Neural Tangent Kernel: Convergence and Generalization"
- Neyshabur et al. (2017) "Exploring Generalization in Deep Networks"

---

## Phase 6: Generative Models

### VAEs
- *Kingma & Welling (2014) "Auto-Encoding Variational Bayes"
- Rezende et al. (2014) "Stochastic Backpropagation and Approximate Inference"

### GANs
- *Goodfellow et al. (2014) "Generative Adversarial Nets"
- *Arjovsky et al. (2017) "Wasserstein GAN"
- Karras et al. (2019) "A Style-Based Generator Architecture for GANs" — StyleGAN

### Diffusion Models
- *Ho et al. (2020) "Denoising Diffusion Probabilistic Models" — DDPM
- *Song et al. (2021) "Score-Based Generative Modeling through SDEs"
- Rombach et al. (2022) "High-Resolution Image Synthesis with Latent Diffusion Models" — Stable Diffusion

### Video & 3D
- Ho et al. (2022) "Video Diffusion Models"
- Mildenhall et al. (2020) "NeRF: Representing Scenes as Neural Radiance Fields"

---

## Phase 7: Advanced Topics

### Continual Learning
- *Kirkpatrick et al. (2017) "Overcoming Catastrophic Forgetting in Neural Networks" — EWC
- Zenke et al. (2017) "Continual Learning Through Synaptic Intelligence"
- Rusu et al. (2016) "Progressive Neural Networks"

### Neural ODEs & Continuous Networks
- *Chen et al. (2018) "Neural Ordinary Differential Equations"
- Dupont et al. (2019) "Augmented Neural ODEs"

### Geometric Deep Learning
- *Bronstein et al. (2021) "Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges"
- Cohen & Welling (2016) "Group Equivariant Convolutional Networks"

### Reinforcement Learning
- Sutton & Barto "Reinforcement Learning: An Introduction" — the textbook
- *Schulman et al. (2017) "Proximal Policy Optimization"
- *Ouyang et al. (2022) "Training Language Models to Follow Instructions with Human Feedback" — RLHF

---

## Phase 8: Research Frontier

### Scaling Laws
- *Kaplan et al. (2020) "Scaling Laws for Neural Language Models"
- *Hoffmann et al. (2022) "Training Compute-Optimal Large Language Models" — Chinchilla

### Mixture of Experts
- *Shazeer et al. (2017) "Outrageously Large Neural Networks: The Sparsely-Gated MoE"
- Fedus et al. (2022) "Switch Transformers: Scaling to Trillion Parameter Models"

### State Space Models
- *Gu et al. (2022) "Efficiently Modeling Long Sequences with Structured State Spaces" — S4
- *Gu & Dao (2023) "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

### Dynamical Systems Perspective
- *Ramsauer et al. (2020) "Hopfield Networks is All You Need" — modern Hopfield
- Hanin & Rolnick (2018) "How to Start Training: The Effect of Initialization and Architecture"
- E (2017) "A Proposal on Machine Learning via Dynamical Systems"

### The Frontier
- Olsson et al. (2022) "In-context Learning and Induction Heads" — mechanistic interpretability
- Elhage et al. (2022) "Toy Models of Superposition"
- Park et al. (2023) "The Geometry of Truth"

---

## Books (Full References)

| Book | Author | Use For |
|------|--------|---------|
| Linear Algebra Done Right | Axler | Rigorous linear algebra |
| Convex Optimization | Boyd & Vandenberghe | Optimization theory |
| Pattern Recognition and ML | Bishop | Classical ML foundations |
| Deep Learning | Goodfellow, Bengio, Courville | Comprehensive DL reference |
| Information Theory, Inference, and Learning | MacKay | Info theory + Bayesian ML |
| Probability Theory: The Logic of Science | Jaynes | Bayesian foundations |
| Reinforcement Learning | Sutton & Barto | RL foundations |
| Geometric Deep Learning | Bronstein et al. | Symmetry and geometry in DL |

---

## How to Read Papers

1. **First pass (10 min):** Read title, abstract, introduction, conclusion. Understand the claim.
2. **Second pass (1 hour):** Read the full paper, skip proofs. Understand the method.
3. **Third pass (4+ hours):** Re-derive the math. Implement the key algorithm. This is where understanding happens.

*"Reading without implementing is like watching someone swim and thinking you can swim."*
