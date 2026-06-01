# MATH 647 — Random Matrix Theory in Machine Learning (Spring 2026)

**Instructor:** Yizhe Zhu (yizhezhu@usc.edu)  
**Schedule:** MWF 1:00–1:50 pm, GFS 112  
**Office Hours:** Friday 12:00–12:50 pm, KAP 464B  
**Course Site:** https://sites.google.com/usc.edu/yizhezhu/teaching/math-647-spring-2026  
**Assignments:** Posted on Brightspace  

## Course Description

Random Matrix Theory (RMT) lies at the intersection of probability, linear algebra, and mathematical physics, and has become a central tool in modern data science and machine learning. This course introduces fundamental methods and concepts of RMT with emphasis on modern machine learning applications, covering both asymptotic and non-asymptotic perspectives. Topics span network analysis, signal processing, and deep learning theory.

## Topics

**Chapter 1 — Asymptotic Theory for Random Matrices**
- Wigner's semicircle law
- Marchenko–Pastur law
- Spectral norm bound with the high moment method
- Stieltjes transform
- Marchenko–Pastur law with relaxed dependence
- Semicircle law with Stieltjes transform
- Local law and eigenvector delocalization

**Chapter 2 — Community Detection in the Stochastic Block Model**
- Matrix concentration inequality
- Davis–Kahan inequality
- Weak and exact recovery with the spectral method
- Leave-one-out analysis for ℓ∞-norm perturbation bound
- Detection threshold
- The non-backtracking operator

**Chapter 3 — Spiked Matrix Models**
- Spiked Wigner model and BBP transition
- Spiked sample covariance model
- Tensor PCA
- Information-theoretic threshold for tensor PCA
- Algorithmic threshold for tensor PCA

**Chapter 4 — Nonlinear Random Matrices and Deep Learning Theory**
- Student–teacher model, training error, and generalization error
- Random feature regression
- Concentration of random kernel matrices
- Spectrum of conjugate kernel in the proportional limit
- Benign overfitting in linear regression
- Double descent
- Neural tangent kernel

**Chapter 5 — High-Dimensional Learning Dynamics**
- Streaming least squares with isotropic data
- Anisotropic least squares and homogenized SGD

## Final Project Paper Options

**Concentration Inequalities**
- Bandeira et al. — Matrix Chaos Inequalities
- Boedihardjo — Injective norm of random tensors
- Hinrichs et al. — Random sections of ellipsoids
- Brailovskaya & van Handel — Extremal random matrices

**Spectral Clustering**
- Zhang — Fundamental Limits
- Carpentier et al. — Phase Transition in SBM
- Tran & Vu — Perturbation bounds

**Spiked Models**
- Guionnet et al. — Non-Linear Wigner
- Feldman — Elementwise-Transformed Spiked
- Li — Algorithmic Phase Transition
- Kothari & Xu — Tensor PCA
- Ding et al. — Rank-One Spikes

**Nonlinear Models**
- Benigni & Paquette — Neural Tangent Kernel
- Kaushik et al. — Empirical Kernel Matrices
- Wang et al. — Nonlinear spiked covariance

**Deep Learning Theory**
- Ghosh & Belkin — Model Size Trade-offs
- Lin et al. — Scaling Laws
- Ba et al. — Low-dimensional Structure
- Dandi et al. — Two-layer Networks
- Wang et al. — Diffusion Models
- Lu et al. — In-context Learning

**High-Dimensional Statistics**
- Bai & Zhou — Covariance matrices
- Rezaei et al. — Synthetic Data Selection
- Green & Romanov — PCA
- Yun & Dudeja — Differentially Private PCA

## Folder Structure

```
MATH647/
├── Lectures/                        # Lecture notes
│   ├── Math_647__RMT_in_ML.pdf      # Full course notes (all chapters)
│   ├── 01_Wigner_Semicircle/
│   ├── 02_Marchenko_Pastur/
│   ├── 03_Stieltjes_Local_Laws/
│   ├── 04_Spectral_Clustering_SBM/
│   ├── 05_Matrix_Concentration/
│   ├── 06_Spiked_Models/
│   ├── 07_Tensor_PCA/
│   ├── 08_Neural_Networks_NTK/
│   ├── 09_Benign_Overfitting/
│   └── 10_High_Dim_Learning/
├── Homework/                        # HW assignments and solutions (Brightspace)
│   ├── HW1/ ... HW5/
├── References/                      # Final project papers by topic
│   ├── Concentration_Inequalities/
│   ├── Spectral_Clustering/
│   ├── Spiked_Models/
│   ├── Nonlinear_Random_Matrices/
│   ├── Deep_Learning_Theory/
│   └── High_Dim_Statistics/
├── Final_Project/                   # Final project (presentations: weeks 16–17)
│   ├── Paper/
│   ├── Presentation/
│   └── Notes/
└── Notes/                           # Personal notes / scratch
```
# MATH647
