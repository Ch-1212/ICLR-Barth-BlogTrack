---
layout: post
title: How does the inductive bias influence the generalization capability of neural networks?
authors: Barth, Charlotte
tags: [overfitting puzzle, overparametrization, memorization, generalization]
---

<style>
    .blue_highlight_mx-e .MathJax{
        background-color: rgba(24, 40, 242, 0.36);
        padding-top: 25px;
        padding-bottom: 25px;
        padding-left: 4px;
        padding-right: 4px;
    }
    .yellow_highlight_mx-e .MathJax{
        background-color: rgba(241, 196, 60, 0.56);
        padding-top: 25px;
        padding-bottom: 25px;
        padding-left: 4px;
        padding-right: 4px;
    }

</style>

<!---
Abstract 2-3 Sätze, worum gehts im Blogpost
Redefluss soll da sein, leicht einzusteigen
--->
This blogposts gives deeper insights on the theoretical question of how inductive biases influence the generalization capability of neural networks. It therefore, analyses the paper [Identity Crisis: Memorization and Generalization under Extreme Overparameterization](https://arxiv.org/pdf/1902.04698.pdf), gives some more background on the underlying paradox and explains the paper's experiments and findings.


## Overfitting Puzzle
<!---
Overparametrization: Memorization/ Generalization Theorie gegen statistische Lerntheorie!! -> Poggio Doktorvater Hauptautor
Je mehr Testdaten, desto besser
Generelles Problem in Graphik erklären!
--->
Neural networks are widly used in machine learning and it is commonly accepted that they outperform most other models. The reasons why these networks perform so well, however, are not well understood. One open question is the **overfitting puzzle**.
It describes the paradox where neural networks contradict the statistical learning theory.

**Statistical learning theory** says that a model should not have more parameters than data (overparametrization) as this leads to a model that is not able to predict well on new unseen data (generalization). During training such an overparametrized model can simply memorize the training data and therefore, performs well on those (overfitting).
Neural networks however (especially deep networks) are typically used in the overparametrized case, where the number of parameters exceed training exapmles. In this regime several common generalization bounds do not apply [Poggio et al. [2017]](https://arxiv.org/pdf/1801.00173.pdf).

In classical learning theory the generalization behavior of a learning system depends on the number of training data n. From this point of view deep learning networks behave as expected: the more training data, the smaller the test error.
A trade-off between the complexity of a model and its fit to the data is assumed [Poggio et al. [2017]](https://arxiv.org/pdf/1801.00173.pdf). The complexity of neural networks is often measured by the number of parameters.
However, neural networks have proved this wrong. Even in cases of extreme overparametrization (e.g. a 5-layer CNN may have 80 million parameters [Zhang et al. [2020]](https://arxiv.org/pdf/1902.04698.pdf)) these networks perform well on unseen data/ are able to generalize. Although this behavior of neural networks is widly used in machine learning, the underlying reasons are still not well understood.
Zhang et al.'s paper aims to get a deeper understanding of this phenomenon. It more specifically, analyses the role of the **inductive bias**. The inductive bias or learning bias results from assumptions a network makes about the nature of the target function and is structural. Therefore, it highly depends on the architecture of a neural network. [Zhang et al. [2020]](https://arxiv.org/pdf/1902.04698.pdf) make several experiments with different types of fully connencted networks (FCN) as well as convolutional neural networks (CNN) to find out which biases apply for these network architectures.

## Experiments
<!--- 
Section: was macht das Paper? Was ist die Grundidee?
Was ist identity mapping
Highlevel, Setup erklären!
paar Ergebnisse vorstellen (z.B. Figure 2,3; Was sehen wir? Graphiken, GIF?, wechsle sieht)
Erklärung, warum wenige Layer besser funktionieren als viele
2 Theoreme erklären? nicht zu speziell
Leslichkeit erhöhen, in dem man interessante Bilder rauspickt, direkt reintut. Reihe 2, Bild 2
Bilder aufteilen in mnist, fashion mnist, andere

Overparametrization funktioniert schlecht! 20 Layer schlechter als 5

Vllt noch Figure 4 erklären

Bilder reduzieren auf Wichtiges!

Spätere Bilder, e.g. 7 nicht mehr so wichtig, aber bestätigt deren Ergebnis. Größere Filter, mehr Parameter
--->

The paper [Identity Crisis: Memorization and Generalization under Extreme Overparameterization](https://arxiv.org/pdf/1902.04698.pdf) uses empirical studies to get a deeper understanding of the theoretical "overfitting puzzle" and how the inductive bias influences this behaviour of overparametrized neural networks. Zhang et al. compare fully connected and convolutional neural networks that learn an identity map through a single data point. Only using one data point for training is an artificial setup that shows the extremest case of overparametrization.
The goal of their study is to determine if a network tends towards memorization (learning a constant function) or generalization (learning the identity function).

Let's have a deeper look at the identity task. This training objective should return the input itself as ouput. This allows for rich visualization and easy evaluation via correlation or MSE.
For linear models this can be done by ensuring that hidden dimensions are not smaller than the input and by setting the weights to the identity matrix in every layer. For models that use [ReLU] (https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) as activation function it has to be considered that all negative values are discarded. This means, these values need to be encoded and recovered after ReLU which can be achieved by using hidden dimensions twice the input size. Negative and positive values are stored seperately and are then, reconstructed.

The experience setup uses the 60k [MNIST digit data set](http://yann.lecun.com/exdb/mnist/). One of these digits is used as the training example whereas the network's are tested on different kind of data: a linear combination of two digits, random digits from MNIST test set, random images from Fashion MNIST, and some algorithmically generated image patterns.

So let's look at some of the results:


{% include 2022-05-03-submission-barth/Figure2_3.html %}


### Fully connected networks (FCN)

The figure shows that for fully connected networks the outputs differ depending on the depth of the network as well as the type of testing data.
Shallower networks seem to integrate random white noise to the output.
The simmilarity of the test data to the training example also influences the behaviour of the model. On test data from the MNIST digit sets, all network architectures perform quite well. For a similar data point the output tends to be similar to the training output whereas, for more different data points the white noise dominates the output. The authors underline this finding with a *theorem*, proven for 1-layer FCNs. 

$$\begin{equation}
    f(x) = \Pi(x) + R \Pi(x)
    \end{equation}$$

It decomposes the test data point x into orthogonal components that are parallel and perpendicular to the training example $\hat{x}$. This can be looked at as a measure to identify the similarity to $\hat{x}$. R is a random matrix. Whichever part of x dominates, determines if the output is more similar to the training output or includes random white noise. 

This can be confirmed by the visualizations of the 1-layer FCN:

<figure>
  <img src="{{ site.url }}/public/images/Figure2_1layer.png" alt="1-layer FCN results" style="width:900px;display: block;margin-left: auto;margin-right: auto;"><figcaption>Visualization of results for 1-layer-FCN</figcaption>
</figure>

This means the inductive bias does neither lead to good generalization nor to memorization. The predictions become more and more random the more unlike the test data point is to the training data.

Deeper networks tend to learn the constant funtion, so there is a strong inductive bias towards the training output regardless of the specific input. This behaviour is similar to a deep ReLU network as seen in the figure.

<figure>
  <img src="{{ site.url }}/public/images/Figure2_compareFCNReLU.png" alt="Comparison results of deep FCN and deep ReLU" style="width:900px;display: block;margin-left: auto;margin-right: auto;"> <figcaption>Comparison results of deep FCN and deep ReLU</figcaption>
</figure>

Zhang et al. conclude that the simpler the network architecture is, the better it generalizes. This can be seen in line with the statistical learning theory as a simpler architecture means less parameters and therefore, less overparametrization.

### Convolutional neural networks (CNNs)

For convolutional neural networks the inductive bias was analysed using the ReLU activation function and testing networks with different depths. The hidden layers of the CNN consist of 5 × 5 convolution filters organized as 128 channels. The networks have two constraints to match the structure of the identity target function. <!--- mehr Tiefe hier? Appendix B3--->

Figure shows the resulting visualizations. It can be seen that for shallow networks the results are quite good, the identity function could be learned. For intermediate depths neither the identity nor the constant function were adapted, the networks function as edge detectors. In contrast, deep networks learn the constant function.

Again the experiments show that the similarity of the test data to the training data point increases the task success.

Nonetheless, the figure shows that CNNs have a better generalization capability than FCNs.


It is important to note that the experiments primarily compare different neural networks within their architecture type and a comparison between FCNs and CNNs can not be seen as fair. CNNs have natural advantages due to sparser networks and structural biases such as local receptive field and parameter sharing that are consistent with the identity task. Looking at overparametrization CNNs have more parameters. To make this clear: A 6-layer FCN contains 3.6M parameters, a 5-layer CNN (with 5x5 filters of 1024 channels) 78M parameters. This can be seen in figure
<!--- add figure with Table 1 appendix: depends a lot of the channels used: in figure 3 128 channels --->
Concluding, CNNs generalize better than FCNs although they have more paraemeters. This follows the studied phenomenon that neural networks resist the statistical learning theory.

## Conclusion
Having read this blog post we hope the concept of the overfitting puzzle is understood and helps to recognize the significance of the study conducted by Zhang et al.. The setup alone is a smart way to approach this topic and allows intuitive interpretation. The authors find that CNNs tend to “generalize” in terms of actually learning the concept of an identity, whereas FCNs are prone to memorization. Within these networks it can be said that the simpler the network architecture is, the better it generalizes.Another observation is that deep CNNs exhibit extreme memorization. Interesting future work would be to analyse the inductive bias for other data types (e.g. sequence data like speech) and compare if the stated theorems also hold for those cases.



<!---
Weglassen: Paper gets a lot of traffic, already cited 50 times. -> sehr theoretisches Paper, Top15


The paper compares two different neural networks: Fully connected network (FCN) and Convolutional neural network (CNN). CNN have two structural constraints:
receptive field of each neuron is limited to a spatially local neighborhood weights are tied and being used across the spatial array

The paper sets up two theorems: one for one-layered FCNs and the other for one-layered CNNs

They find that CNNs tend to “generalize” in terms of actually learning the concept of an identity, whereas FCNs are prone to memorization. The authors also present results under various different settings such as changing the filter size or the number of hidden channels of CNNs. The conclusion is that the simpler the network architecture is, the better it generalizes. Another observation is that deep CNNs exhibit extreme memorization.

We are primarily studying the comparisons of different architecture choices within each of the FCNs and CNNs family. But a fair comparison between FCNs and CNNs that controls the natural advantage of sparser connections in CNN could indeed be derived from our data. In particular, we have added a table summarizing the parameter counts for all the architectures used in this paper (see Table 1 in the appendix of the updated paper). From the table, we can see that a 6-layer FCN contains 3.6M parameters, while a 5-layer CNN with 5x5 filters of 1024 channels contains 78M parameters, an order of magnitude more than the FCN. Yet the CNN learns the identity function (Fig. 10a) while the FCN does not (Fig. 2). A more similar comparison is with 6 or 7-layer CNNs with 7x7 filters of 128 channels, with 3.2M and 4.0M parameters, respectively. The conclusion is the same. This is consistent with recent observations in the deep learning community that overparameterization, especially when measured by parameter counting, is not necessarily at odds with generalization

We added experiment results on CIFAR-10 images to the updated version of the paper (please see Appendix O). The observations are consistent with the MNIST case. Since the design of our task is to propagate all the input pixels to the outputs, the semantic complexity of the inputs matters less here than in classification tasks. We choose image data because it is intuitive to visualize and interpret (a primary goal of this paper). But we firmly agree that other domains of data and their associated architectures are valuable topics to study. For sequence inputs such as speech and language data, because both the structures in the data and the natures of the commonly used network architectures (e.g. RNNs and Transformers) are drastically different, the experiment framework, analysis and interpretations of memorization need to be completely redesigned.

big picture and takeaways of the paper: there are many interesting observations from our results that could benefits the practitioners when designing models for new tasks. i) parameter counting does not strongly correlate with the generalization performance, but the structural bias of the model does. For example, being equally overparameterized, ii) training a very deep model without residual connections might be prune to memorization; while iii) adding more feature channels / dimensions is much less likely to overfit. As a result, if one is to increase the number of parameters of an existing model (maybe hoping to get smoother optimization dynamics, or to prepare for more training data), it is better to first try to increase the hidden dimension before tuning the depth, unless the nature of the data changes. Our new results, suggested by Reviewer 3, show that iv) even with explicit identity skip connections as in resnets, the amount of prediction noises still grows with the network depth.

<!--- Interesting finding that depth is worse than bigger hidden dimension! INCLUDE THAT? --->

<!---
aim to gain deeper understanding of the mythical inductive bias of overparameterized neural networks that seem to magically avoid overfitting and generalize well. We approach this theoretical question using empirical studies. In particular, we design a very specific and controlled experiment setting where we <!--- have clear and unambiguous definition of memorization and generalization ---> 
<!--- 
. The extensive set of experiments are designed around this setting to reveal the inductive bias under different architectural choices. Our results show that deep networks do not always have magical inductive biases that help to generalize well. CNNs have structural biases such as local receptive field and parameter sharing that are consistent with our task, thus demonstrate stronger generalization capability than FCNs. But deeper CNNs still bias towards memorization. We believe our experiments are important contributions to the endeavor towards full understanding of the generalization behaviors of deep learning.

 Yet, in my opinion, the idea of using identity as training objective is even bigger contribution that the study itself.
--->

## References
[Zhang, C., Bengio, S., Hardt, M., Mozer, M. C., & Singer, Y. (2019). Identity crisis: Memorization and generalization under extreme overparameterization. arXiv preprint arXiv:1902.04698.](https://arxiv.org/pdf/1902.04698.pdf)

[Poggio, T., Kawaguchi, K., Liao, Q., Miranda, B., Rosasco, L., Boix, X., ... & Mhaskar, H. (2017). Theory of deep learning III: explaining the non-overfitting puzzle. arXiv preprint arXiv:1801.00173.](https://arxiv.org/pdf/1801.00173.pdf)

[Zhang, C., Bengio, S., Hardt, M., Mozer, M. C., & Singer, Y. (2019). Identity crisis: Memorization and generalization under extreme overparameterization. arXiv preprint arXiv:1902.04698.](https://arxiv.org/pdf/1902.04698.pdf)


