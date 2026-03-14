Generative AI 2025/2026 - 2nd Semester

Project 1: ArtBench Generative Modeling

## **1 Introduction**

We present here the first practical project, part of the students’ evaluation process of the Generative AI course of the University of Coimbra. This work is to be done autonomously by a group of **two students** . The quality of your work will be judged as a function of the value of the technical work, the written description, and the public defence.

All sources used to perform the work, including code, pretrained components, external repositories, and **Generative AI tools** , must be clearly identified. If generative AI systems, e.g., code assistants or large language models, are used during the development of the work, their use must be explicitly acknowledged and described. Students remain fully responsible for the correctness, originality, and integrity of all submitted material.

The **deadline** for delivering the work is **19 Abril** of 2026 via Inforestudante.

The document may be written in Portuguese or in English and the report must follow the **Springer LNCS (Lecture Notes in Computer Science) format** . The official LaTeX template can be obtained from the Springer website[1] , and an online version is also available on Overleaf[2] .

The **written report** is limited to **12 pages** using the LNCS template excluding references. The report should be well structured and include the following elements: a general **introduction** , a **description of the problem** , the **methodology and approach** , the **experimental setup** , an **analysis of the results** , and a **conclusion** . The final mark is given individually. Plagiarism will not be allowed and, if detected, it implies failing the course. While doing the work and when submitting it, you should pay particular attention to:

- clear description of the adopted approach;
- description of model architectures and training details;
- reproducible experiment protocol, including hyperparameters and random seeds;
- quantitative evaluation using appropriate metrics for generative models.

> 1 `https://www.springer.com/gp/computer-science/lncs/ conference-proceedings-guidelines`

> 2 `https://www.overleaf.com/latex/templates/springer-lecture-notes-in-computer-science/ kzwwpvhwnvfj`

2

## **2 Problem Statement**

In this project, the goal is not classification but **image generation** . Students must learn a generative model from artwork images and produce new plausible samples that match the data distribution.

The target domain is the **ArtBench** dataset, composed of paintings grouped by artistic style. This is a challenging setting because models must balance visual fidelity, diversity, and style consistency. Unlike discriminative tasks, quality must be assessed both qualitatively (sample inspection) and quantitatively (distribution-level metrics).

Students are expected to compare multiple families of generative models and analyze trade-offs in stability, sample quality, and computational cost.

## **3 Objective**

The main objective is to design, train, and evaluate generative models on ArtBench under a controlled protocol.

- Build a full generative pipeline: data loading, preprocessing, training, sampling, and evaluation.
- Use the **provided subset of the training data** , for model development and comparison.
- Compare qualitatively and quantitatively (Section 3.2) at least the following model families:

  - **Autoencoders** : at least VAE-type model;
  - **GANs** : at least one DCGAN-based model;
  - **Diffusion models** : at least one diffusion-based model.
- Select the best-performing approach on the subset and retrain/evaluate it on the **full training dataset** .

For each model family we have provided a starting reference pack at the end of this document and more information can be found on online references such as: the paper for artbench[3] , huggingface[4] or github[5] . Exploring

> 3 `https://arxiv.org/pdf/2206.11404`

> 4 `https://huggingface.co/docs/diffusers/v0.37.0/en/tutorials/basic_ training#training-configuration`

> 5 `https://github.com/facebookresearch/pytorch_GAN_zoo`

3

and presenting results of additional methods is encouraged and may be considered as extra work that can compensate for missing points on the base requirements of the project.

## **3.1 Dataset**

This project uses the **ArtBench-10 dataset**[6] [7]. ArtBench-10 is a curated dataset of artworks designed to benchmark generative models in the visual arts domain. It contains images multiple artistic styles and was specifically introduced to evaluate generative models such as VAEs, GANs, and diffusion models on structured artistic data.

During development and experimentation, students should initially train and tune their models using the **provided 20% subset of the training data** . This allows faster iteration when exploring architectures, hyperparameters, and training strategies. Once a suitable configuration has been identified, the selected model should then be trained on the **full training dataset** for the final evaluation.

- Number of classes: **10 artistic styles**
- Image resolution: 32 _×_ 32 **RGB**
- Dataset splits: **50000 training samples**

A starting codebase is provided that allows loading the full dataset and the provided subset, visualizing samples, and creating data loaders ready for training.

## **3.2 Evaluation Metrics and Protocol**

For evaluation of the different models on the same dataset you should provide visual outputs for the qualitative evaluation, provide figures on the report that allows to see the different outputs of each model. The mandatory quantitative metrics are **FID** (Fr´echet Inception Distance) and **KID** (Kernel Inception Distance), which are commonly used to evaluate generative image models. You have to follow this evaluation protocol:

- For each evaluated generative model, generate **5000 samples** .
- Sample **5000 real images** from the **ArtBench dataset** .

> 6 `https://www.kaggle.com/datasets/alexanderliao/artbench10`

4

- Compute both **FID** and **KID** between the generated and real samples.

## **FID computation:**

- FID must be computed using the full sets of **5000 generated** and **5000 real** images.

## **KID computation:**

- KID must be computed using random subsets.
- Use **50 subsets of size 100** sampled from the generated and real image sets.
- Report the **mean and standard deviation of KID** across the subsets.

For Statistical robustness and since this is a stochastic process, we need to attend to some extra requirements for evaluation. The full evaluation protocol must be repeated using **different random seeds** , with a minimum of **10 repetitions** per fixed model configuration. Report the final results as:

- **Mean** _±_ **standard deviation of FID**
- **Mean** _±_ **standard deviation of KID**

For fairness, preprocessing, sample count, and evaluation code path must remain fixed across all compared models.

## **4 Conclusion**

This project emphasizes rigorous experimental methodology for generative modeling: controlled data protocols, fair cross-model comparison, and statistical reporting across seeds. The final discussion should clearly connect model design decisions to observed quality/diversity trade-offs and metric behavior.

Good luck!

## **Referˆencias**

- [1] P. Dhariwal and A. Nichol. Diffusion models beat gans on image synthesis. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2021.

5

- [2] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. C. Courville. Improved training of wasserstein gans. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2017.
- [3] J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. In _Advances in Neural Information Processing Systems (NeurIPS)_ , 2020.
- [4] T. Karras, M. Aittala, S. Laine, E. H¨ark¨onen, J. Hellsten, J. Lehtinen, and T. Aila. Training generative adversarial networks with limited data. _arXiv preprint arXiv:2006.06676_ , 2020.
- [5] T. Karras, S. Laine, and T. Aila. A style-based generator architecture for generative adversarial networks. _arXiv preprint arXiv:1812.04948_ , 2018.
- [6] D. P. Kingma and M. Welling. Auto-encoding variational bayes. _arXiv preprint arXiv:1312.6114_ , 2013.
- [7] P. Liao, X. Li, X. Liu, and K. Keutzer. The artbench dataset: Benchmarking generative models with artworks. _arXiv preprint arXiv:2206.11404_ , 2022.
- [8] A. Radford, L. Metz, and S. Chintala. Unsupervised representation learning with deep convolutional generative adversarial networks. _arXiv preprint arXiv:1511.06434_ , 2015.
- [9] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. Highresolution image synthesis with latent diffusion models. In _IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , 2022.
- [10] K. Sohn, H. Lee, and X. Yan. Learning structured output representation using deep conditional generative models. _arXiv preprint arXiv:1506.05517_ , 2015.

6
