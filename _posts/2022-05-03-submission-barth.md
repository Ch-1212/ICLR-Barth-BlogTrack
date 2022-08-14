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
Deep neural networks have long become the golden standard for machine learning and are applied to many different use cases. Their ability to generalize from training data is widly used but not well understood.
This blogposts gives deeper insights on the theoretical question why neural networks generalize, more concretely, how inductive biases influence the generalization capability of neural networks. It therefore, analyses the paper [Identity Crisis: Memorization and Generalization under Extreme Overparameterization](https://arxiv.org/pdf/1902.04698.pdf), gives some more background on the underlying paradox and explains the paper's experiments and findings. It also shows how practitioners can benefit from the theoretical findings and how they can be used when designing new models.


## Overfitting Puzzle
<!---
Overparametrization: Memorization/ Generalization Theorie gegen statistische Lerntheorie!! -> Poggio Doktorvater Hauptautor
Je mehr Testdaten, desto besser
Generelles Problem in Graphik erklären!
--->
Neural networks are widly used in machine learning and it is commonly accepted that they outperform most other models. The reasons why these networks perform so well, however, are not well understood. One open question is the **overfitting puzzle**.
It describes the paradox where neural networks contradict the statistical learning theory.

**Statistical learning theory** says that a model should not have more parameters than data (overparametrization) as this leads to a model that is not able to predict well on new unseen data (generalization). During training such an overparametrized model can simply memorize the training data and therefore, performs well on those (overfitting).
Neural networks however (especially deep networks) are typically used in the overparametrized case, where the number of parameters exceed training examples. In this regime several common generalization bounds do not apply [Poggio et al. [2017]](https://arxiv.org/pdf/1801.00173.pdf).

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
For linear models this can be done by ensuring that hidden dimensions are not smaller than the input and by setting the weights to the identity matrix in every layer. For the convolutional layer only the center of the kernel is used and all other values are set to zero. It therefore, simulates a 1 x 1 convolution which functions as a local identity function. For deeper models that use [ReLU] (https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) as activation function it has to be considered that all negative values are discarded. This means, these values need to be encoded and recovered after ReLU which can be achieved by using hidden dimensions twice the input size. Negative and positive values are stored seperately and are then, reconstructed.

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

Zhang et al. conclude that the more complex the network architecture is, it is more prone to memorization. This can be seen in line with the statistical learning theory as a more complex architecture means more parameters and therefore, more overparametrization.

### Convolutional neural networks (CNNs)

For convolutional neural networks the inductive bias was analysed using the ReLU activation function and testing networks with different depths. The hidden layers of the CNN consist of 5 × 5 convolution filters organized as 128 channels. The networks have two constraints to match the structure of the identity target function.

Figure shows the resulting visualizations. It can be seen that for shallow networks the results are quite good, the identity function could be learned. For intermediate depths neither the identity nor the constant function were adapted, the networks function as edge detectors. In contrast, deep networks learn the constant function. Wether the model learns the identity or the constant function, both outcomes reflect inductive biases as no specific structure was been given by the task.

{% include 2022-05-03-submission-barth/CNNs_intermedLayers.html %}

The evolution of the output can be better understood when looking at the status of the prediction in the hidden layers of the CNN. Since CNNs - unlike FCNs - preserve the spatial relations between neurons in the intermediate layers, these layers can be visualized. Figure shows a randomly initialized 20-layer CNN compared to different depths of trained CNNs. Random convolution grdually smooths out the input data and after around eight layers the shapes are lost. It can be seen that training the networks lead to different results. The 7-layer CNN functions quite well and ends up with an identity function of the input images whereas the results of the 14-layer CNN are more blurry. For the 20-layer trained CNN it can be seen that it first behaves similar to the randomly initialized CNN, wiping out the input data, but keeps the shapes a bit longer. In the last three layers it is rendering the constant function of the training data and outputting 7 for whichever input.

As for the FCNs, the experiments show that the similarity of the test data to the training data point increases the task success.

Nonetheless, the figures show that CNNs have a better generalization capability than FCNs.

Zhang et al. did further experiments with different feature channel numbers and dimensions. Interestingly, they found that increasing the hidden dimensions/ adding channels is way less prune to overfitting than adding depth. This should be taken into account when designing new models: if one is to increase the number of parameters of an existing model (maybe hoping to get smoother optimization dynamics, or to prepare for more training data), it is better to first try to increase the hidden dimension before tuning the depth, unless the nature of the data changes.

Another aspect that influences the inductive bias is model initialization. Especially for networks with few channels the difference between random initialization and the converged network is extreme [Frankle et al. (2018)](). This can be explained in the following way: In a regime of random initialization with only few channels, the initialization has not enough changeability to even out wrong choices. Therefore, the networks are more likely to converge to non-optimal extrema. Having more channels evens out this problem as more parameters help smooth out the decisions.

## General findings

It is important to note that the experiments primarily aim to compare different neural networks within their architecture type and a comparison between FCNs and CNNs can not be seen as fair. CNNs have natural advantages due to sparser networks and structural biases such as local receptive field and parameter sharing that are consistent with the identity task. Looking at overparametrization CNNs have more parameters. To make this clear: A 6-layer FCN contains 3.6M parameters, a 5-layer CNN (with 5x5 filters of 1024 channels) 78M parameters. This can be seen in figure

<figure>
  <img src="{{ site.url }}/public/images/#param.gif" alt="Comparison # parameters for FCN and CNN" style="width:900px;display: block;margin-left: auto;margin-right: auto;"> <figcaption>Number of parameters for FCN and CNN</figcaption>
</figure>

Concluding, CNNs generalize better than FCNs although they have more parameters. This follows the studied phenomenon that neural networks resist the statistical learning theory.

To conclude, the main findings of the paper:
i) parameter counting does not strongly correlate with the generalization performance, but the structural bias of the model does. For example, being equally overparameterized,
ii) training a very deep model without residual connections might be prune to memorization; while 
iii) adding more feature channels / dimensions is much less likely to overfit.

## Conclusion

Having read this blog post we hope the concept of the overfitting puzzle is understood and helps to recognize the significance of the study conducted by Zhang et al.. The setup alone is a smart way to approach this topic and allows intuitive interpretation. The authors find that CNNs tend to “generalize” in terms of actually learning the concept of an identity, whereas FCNs are prone to memorization. Within these networks it can be said that the simpler the network architecture is, the better it generalizes. Another observation is that deep CNNs exhibit extreme memorization. It would have been interesting to analyse the inductive bias for other data types (e.g. sequence data like speech) and compare if the stated theorems also hold for those cases.

In summary, Zhang et al. conducted interesting studies that helped the machine learning community to get a deeper understanding of the inductive bias. From their results concrete guidance for practitioners can be derived that can help design models for new tasks.


<!--- CNN have two structural constraints:
receptive field of each neuron is limited to a spatially local neighborhood weights are tied and being used across the spatial array

The paper sets up two theorems: one for one-layered FCNs and the other for one-layered CNNs

They find that CNNs tend to “generalize” in terms of actually learning the concept of an identity, whereas FCNs are prone to memorization. The authors also present results under various different settings such as changing the filter size or the number of hidden channels of CNNs. The conclusion is that the simpler the network architecture is, the better it generalizes. Another observation is that deep CNNs exhibit extreme memorization.

We added experiment results on CIFAR-10 images to the updated version of the paper (please see Appendix O). The observations are consistent with the MNIST case. Since the design of our task is to propagate all the input pixels to the outputs, the semantic complexity of the inputs matters less here than in classification tasks. We choose image data because it is intuitive to visualize and interpret (a primary goal of this paper). But we firmly agree that other domains of data and their associated architectures are valuable topics to study. For sequence inputs such as speech and language data, because both the structures in the data and the natures of the commonly used network architectures (e.g. RNNs and Transformers) are drastically different, the experiment framework, analysis and interpretations of memorization need to be completely redesigned.

. The extensive set of experiments are designed around this setting to reveal the inductive bias under different architectural choices. Our results show that deep networks do not always have magical inductive biases that help to generalize well. CNNs have structural biases such as local receptive field and parameter sharing that are consistent with our task, thus demonstrate stronger generalization capability than FCNs. But deeper CNNs still bias towards memorization. We believe our experiments are important contributions to the endeavor towards full understanding of the generalization behaviors of deep learning.

--->

## References

$$\printbibliography[heading=bibintoc,title={References}]

[Zhang, C., Bengio, S., Hardt, M., Mozer, M. C., & Singer, Y. (2019). Identity crisis: Memorization and generalization under extreme overparameterization. arXiv preprint arXiv:1902.04698.](https://arxiv.org/pdf/1902.04698.pdf)

[Poggio, T., Kawaguchi, K., Liao, Q., Miranda, B., Rosasco, L., Boix, X., ... & Mhaskar, H. (2017). Theory of deep learning III: explaining the non-overfitting puzzle. arXiv preprint arXiv:1801.00173.](https://arxiv.org/pdf/1801.00173.pdf)

[Frankle, J., & Carbin, M. (2018). The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635.](https://arxiv.org/pdf/1803.03635.pdf)


