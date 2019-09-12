---
layout: post
comments: true
title:  "The Attacks and Defenses of the Deep Neural Network"
categories: [notes]
tags: [deep learning]
---


{: class="table-of-content"}
* TOC
{:toc}

## Adversarial Attacks, Detection and Defenses
### Attacks

1. Fast Gradient Sign Method(FGSM)

2. Iterative Gradient Sign Method

3. DeepFool

4. Carlini's method

5. <a name="circumvent-defense"></a>[Obfuscated Gradients Give a False Sense of Security: Circumventing Defenses to Adversarial Examples](https://www.dropbox.com/s/ptlfweasyq6ieuq/Obfuscated%20Gradients.pdf?dl=0) (ICML2018)

	Optimization-based attacks generate adversarial examples using gradients obtained through backpropagation. The authors identify obfuscated gradients as an element of many defenses that provides a facade of security by causing the attacker’s gradient descent to fail. Obfuscated gradients come in at least three forms: **shattered gradients**, **stochastic gradients**, and **vanishing/exploding gradients**.

	A defense uses **shattered gradients** when it introduces non-differentiable operations, numeric instability, or otherwise causes the attacker’s gradient signal to be incorrect. A defense uses **stochastic gradients** when the inputs to the classifier are randomized or a stochastic classifier is used, resulting in a different gradient each time it is evaluated. **Vanishing/exploding gradients** are issues encountered in training some networks in which the gradient grows or shrinks exponentially during backpropagation. Some defenses involve multiple iterations of neural network evaluation. This can be viewed as evaluating one very deep neural network, which obfuscates the gradient signal by forcing the vanishing/exploding gradient problem.



Based on the adversary's knowledge, the attack can also be classified into different types:

**white-box attacks(perfect knowledge)**: The adversary has complete knowledge of both the classifier and the detector.

**black-box attacks(limited knowledge)**: The attacker knows what type of defense is in place but does not know its parameters and does not have direct access to the model.

**grey-box attacks(zero-knowledge)**: The adversary has access to the model architecture and the model parameters, but is unaware of the defense strategy that is being used.

**oblivious attacks(zero-knowledge)**: The adversary does not know the existence of such defense and only generates adversarial examples that aim to maximize the prediction error of the classifier.


### Detection
1. [ON DETECTING ADVERSARIAL PERTURBATIONS](https://www.dropbox.com/s/mrogs9omkszw3ar/ON%20DETECTING%20ADVERSARIAL%20PERTURBATIONS.pdf?dl=0) (ICLR 2017)  [(*Defeated*)](#defeat-detection) 
   
    They proposed a detector network for detecting adversarial examples as an auxiliary network of the original network architecture. The detector network produces the probability of the input being adversarial. The authors also admits that an adversary can adapt to the detector, which is termed as *dynamic adversary*. Their solution is to use the adversarial training by generating adversarial examples on-the-fly and injecting them back to the training dataset.


2. [MagNet: a Two-Pronged Defense against Adversarial Examples](https://www.dropbox.com/s/cnxbxbdvgyneoc4/MagNet.pdf?dl=0) (ACM CCS 2017)  [(*Defeated*)](https://www.dropbox.com/s/oozg8i3fxawoy37/MagNet%20and%20%E2%80%9CEfficient%20Defenses%20Against%20Adversarial%20Attacks%E2%80%9D.pdf?dl=0) 
   
   In MagNet, the detector is used to detect adversarial examples that are far from the boundry of the normal input manifold. They first train an autoencoder on normal examples only as building blocks. Then they measured the probability divergence between the softmax output of the input images and restructed images after the autoencoder. The adversarial exmaples are expected to have larger divergence. In their implementation, they created a large number of autoencoders to defend greybox attacks.


3. [SafetyNet: Detecting and Rejecting Adversarial Examples Robustly](https://www.dropbox.com/s/k6umulg984qor0d/safetynet.pdf?dl=0)(ICCV 2017)
   
    They proposed the SafetyNet, which uses the discrete codes computed from individual in particular late stage ReLUs layers and feed them to an RBF-SVM classifier to detect adversarial examples. They claim that while the detector is very effective at detecting adversarial examples, the attack on the detetor will mostly fail even though it is discovered.


4. [On the (Statistical) Detection of Adversarial Examples](https://www.dropbox.com/s/1uj562alhl4av2e/On%20the%20%28Statistical%29%20Detection%20of%20Adversarial%20Examples.pdf?dl=0)  [(*Defeated*)](#defeat-detection) 
   
    They employed a statistical test to distinguish adversarial examples from the normal data. They showed that few as 50 adversarial examples per class are sufficient to identify a measurable difference between adversarial distributions and legitimate ones. In addition, they also showed the method to add an outlier class to the original model and classify the adversarial examples as the outliers.


5. [Detecting adversarial samples from artifacts](https://www.dropbox.com/s/izr1cku3m5pjiad/Detecting%20Adversarial%20Samples%20from%20Artifacts.pdf?dl=0)  [(*Defeated*)](#defeat-detection) 
   
    They showed that adversarial patterns can be detected with two features: kernel density estimates in the subspace of the last hidden layer and the Bayesian neural network uncertainty estimates. They claims that the density estimates are generally decreased while the uncertainty is increased when the adversarial examples are generated.  Designing a classifier that uses these two features as input can effectively detect adversarial examples.


6. [Early Methods for Detecting Adversarial Images](https://www.dropbox.com/s/6ek0tap2oy8nhq2/EARLY%20METHODS%20FOR%20DETECTING.pdf?dl=0)  [(*Defeated*)](#defeat-detection) 
   
    They showed that the adversarial images tend to have larger and higher variance for low-ranked componts after being whitened by PCA. Images are classified as clean vs adversarial by fitting two gaussians to use in a likelihood comparison, one for clean examples and another for adversarial.


7. [Feature Squeezing](https://www.dropbox.com/s/0190flflq07or58/Feature%20Squeezing.pdf?dl=0) (NDSS 2018)
   
    The model is evaluated on both the original input and the input after being pre-processed by feature squeezers. If the output difference between the two exceeds a certain threshold level, the input is detected as an adversarial example.


7. [PixelDefend: Leveraging Generative Models to Understand and Defend against Adversarial Examples](https://www.dropbox.com/s/a5rn38pac79ioo0/PixelDefend.pdf?dl=0) (ICLR 2018)[(*Defeated*)](#circumvent-defense) 
   
    The authors describe a method for detecting adversarial examples by measuring the likelihood in terms of a generative model of an image. Furthermore, the authors prescribe a method for cleaning an adversarial image through employing a generative model before applying the classifier. The authors demonstrate some success in restoring images that have been adversarially perturbed with this technique.


8. [Towards Robust Detection of Adversarial Examples](https://www.dropbox.com/s/pl570sikhgwjle3/Towards%20Robust%20Detection%20of%20Adversarial%20Examples.pdf?dl=0) (NIPS 2018)
   
    This paper proposes a combination of two modifications to make neural networks robust to adversarial examples: (1) reverse cross-entropy training allows the neural network to learn to better latent representations that distinguish adversarial examples from normal one, as opposed to standard cross-entropy training, and (2) a kernel-density based detector detects whether or not the input appears to be adversarial, and rejects the inputs that appear adversarial. 
    

### Defenses
1. [EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES](https://www.dropbox.com/s/eu42hxyyjznvbbk/explaining%20and%20harnessing%20adversarial%20examples.pdf?dl=0) (ICLR 2015)

    They proposed adversarial training:  They generated adversarial examples in the every step of training and inject them into the training set. It has been shown that this approach has some limitations — in particular, this kind of defence is less effective against black-box attacks than white-box attacks in which the adversarial images are generated using a different model. This is due to gradient masking, i.e., in these kind of defences, a perturbation in the gradients is introduced, making the white box attacks less effective, but the decision boundary remains mostly unchanged after the adversarial training. 


2. [ENSEMBLE ADVERSARIAL TRAINING ATTACKS AND DEFENSES](https://www.dropbox.com/s/jisibaemllpwfn6/ensemble%20adversarial%20training.pdf?dl=0)(ICLR 2018)

    The paper proposes a modification to adversarial training, in which the generation of the adversarial examples is decoupled from the parameters of the model being trained. This is achieved by drawing the adversarial samples from pre-trained models, which are then added to each batch or used to replace part of the non-adversarial images in the batch.


3. [Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks](https://www.dropbox.com/s/ph305zf1uootmme/Distillation%20as%20a%20Defense%20to%20Adversarial%20Perturbations%20against%20Deep%20Neural%20Networks.pdf?dl=0) (IEEE S&P 2016)

4. [MagNet: a Two-Pronged Defense against Adversarial Examples](https://www.dropbox.com/s/cnxbxbdvgyneoc4/MagNet.pdf?dl=0) (ACM CCS 2017)[(*Defeated*)](https://www.dropbox.com/s/oozg8i3fxawoy37/MagNet%20and%20%E2%80%9CEfficient%20Defenses%20Against%20Adversarial%20Attacks%E2%80%9D%20are%20not%20robust.pdf?dl=0) 
   
   In Magnet, a reformer finds an example near the manifold that can approximates the input image. They also uses autoencoders as reformer. Whenthe input is normal, the autoencoder will reconstruct the image and then the classification result will not change much. However, when the image is adversarial, the autoencoder is expected to output an example that approximates the adversarial example but is closer to the manifold of the normal examples. In this way, the adversarial examples are reformed to be normal.
   
5. [DEEPCLOAK: MASKING DEEP NEURAL NETWORK MODELS FOR ROBUSTNESS AGAINST ADVERSARIAL SAMPLES](https://www.dropbox.com/s/xatabn4ynzpr00w/DEEPCLOAK.pdf?dl=0) (ICLR 2017 workshop)

   

6. [COUNTERING ADVERSARIAL IMAGES USING INPUT TRANSFORMATIONS](https://www.dropbox.com/s/9n5iyshgaa57lcy/countering-adversarial-images-using-input-transformations.pdf?dl=0) (ICLR 2018)[(*Defeated*)](#circumvent-defense) 

   

7. [DEFENSE-GAN: PROTECTING CLASSIFIERS AGAINST ADVERSARIAL ATTACKS USING GENERATIVE MODELS](https://www.dropbox.com/s/xjaxt7c5j7tnshv/defense-gan.pdf?dl=0) (ICLR 2018)[(*Defeated*)](#circumvent-defense) 

   

8. [Feature Denoising for Improving Adversarial Robustness](https://www.dropbox.com/s/gdcn0kdem70a8t8/Feature%20Denoising%20for%20Improving%20Adversarial%20Robustness.pdf?dl=0) (CVPR 2019)

### Bypassing the Detection
1. <a name="defeat-detection"></a>[Adversarial Examples Are Not Easily Detected: Bypassing Ten Detection Methods](https://www.dropbox.com/s/b67nwxymp5xy6ne/Adversarial%20Examples%20Are%20Not%20Easily%20Detected.pdf?dl=0) (2017 ACM Workshop on Artificial Intelligence and Security, 2017. Finalist, Best Paper) 

    Carlini and Wagner study the effectiveness of adversarial example detectors as defense strategy and show that most of them can by bypassed easily by known attacks. Specifically, they consider a set of adversarial example detection schemes, including neural networks as detectors and statistical tests. After extensive experiments, the authors provide a set of lessons which include:

    Randomization is by far the most effective defense (e.g. dropout).
    Defenses seem to be dataset-specific. There is a discrepancy between defenses working well on MNIST and on CIFAR.
    Detection neural networks can easily be bypassed.
    Additionally, they provide a set of recommendations for future work:

    For developing defense mechanism, we always need to consider strong white-box attacks (i.e. attackers that are informed about the defense mechanisms).
    Reporting accuracy only is not meaningful; instead, false positives and negatives should be reported.
    Simple datasets such as MNIST and CIFAR are not enough for evaluation.

### Theory

## Backdoor Attacks and defenses
### Attacks

### Defenses

## Other Poisoning Attack Methods
### [Clean-Label Poisoning Attacks](https://arxiv.org/pdf/1804.00792.pdf)(NeurIPS2018)
This [blog](https://secml.github.io/class11/) explains very well.
