---
layout: post
comments: true
title:  "Meta-learning: Learning to Learn"
categories: [notes]
tags: [deep learning]
---

{: class="table-of-content"}
* TOC
{:toc}

## What is Meta-Learning?
In [Sachin Ravi and Hugo Larochelle](https://openreview.net/pdf?id=rJY0-Kcll)'s work, they propose that different from the traditional machine learning setting, where we usually split $$D​$$ so that we optimize parameters $$\theta​$$ on a training set $$D_{train}​$$ and evaluate its generalization on the test set $$D_{test}​$$, in meta-learning, we are dealing with meta-sets $$\mathscr{D}​$$ containing multiple regular datasets, where each $$D \in \mathscr{D}​$$ has a split of $$D_{train}​$$ and $$D_{test}​$$ (One dataset $$D​$$ is considered as one data sample).



A k-shot, N-class classification task is described as in the training dataset $$D$$, there are k labeled examples for each of N classes. In few shot learning, k is usually a very small number, e.g., one.



In meta-learning, we have different meta-sets,  $$\mathscr{D}_{train}$$ ,  $$\mathscr{D}_{validation}$$  and  $$\mathscr{D}_{test}$$. We normally use $$\mathscr{D}_{train}$$ to train a meta-learner, which aims to take a one of its training sets $$D_{train}$$ and generate a classifier(the learner) that can have a good performance on its corresponding test set ($$\mathscr{D}_{test}$$). Then $$\mathscr{D}_{validation}$$ is used for the hyper-parameter search and finally the method is evaluated on $$\mathscr{D}_{test}$$.