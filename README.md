 ## Deep Convolutional Generative Adversarial Network ##
 
Generative Adversarial Networks (GANs) is an unsupervised learning model to generate new data that are similar to the original data. For example, using MINST data (images of handwritten digits) to generate new images of handwritten digits that is not in the original dataset.
<br/>
<br/>
Another application is the generation of very realistic images of fake people that do not exist.
<br/>
<br/>
Generative Adversarial Networks (DCGANs) are GANs that uses the deep convolutional neural networks for the architecture of the GAN.
<hr/>

GANs consists of 2 parts, a generative network and a discriminatory network. The generative network consists of a generator model, in this case a deep convoutional model that generates images based on a random input. This image generated is known as the generated image or the fake image. This fake image and the real image in the original dataset are then fed into the discriminatory network, consisting of a discriminator model. This discriminator model then tries to classify between the fake images and the real images.
<br/>
<br/>
The main objective of the generator model is to generate an image such that the discriminator model will classify the generated images as real. Thus, the loss function of the generated model is the binary crossentropy, which is the negative mean of the logarithm of the probability output of the generated images from the discriminator model against the class of the real image. In other words, it is trying to maximise the discriminator loss.
<br/>
<br/>
The main objective of the discrimator model is to classify the real images as real and the generated images as fake. Thus, the loss function of the discriminator model is a normal binary crossentropy of the predicted probabilities and its actual class.
<hr/>