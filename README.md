![Tests on latest tensorflow version](https://github.com/jian01/GiaNLP/actions/workflows/tests.yml/badge.svg?branch=main) [![codecov](https://codecov.io/gh/jian01/GiaNLP/branch/main/graph/badge.svg)](https://codecov.io/gh/jian01/GiaNLP)

# GiaNLP

## What is this?

GiaNLP is a library created for the fraud team at Mercadolibre for building, training and deploying NLP models with keras fast.

It's main features are:
* Easy text representation integration: The goal of the library is to plug in costless text representations to a vanilla keras model.
* Easy serialization and deserialization of models: it is easy to transform a complex model to a byte array and then load it for inference or continue the training.
* Train and inference with text: It is not needed to know how to preprocess the texts or in which order, the model encapsulates all.
* Keras-like API and support for Keras interaction via the functional API: All models can receive a keras layer and get it's output like Keras would.
* Arbitrary graph chaining for model architectures: This models can be chained in any way keras models/layers could be chained.
* Possibility of stop using the library at any point of the model: You can use the library in any parts of the graph you want and then chain it to a classic keras model, using the library just for the parts that are built with it.
* Default usage of generators for training and prediction: All texts are fed to the model using generators so preprocessed texts always fit in ram and you don't have to worry about programming your generator.
* Sklearn-like experience (hopefully): Although we are far from that the goal of the library is to have an sklearn-like experience. We wan't to handle multiple object types in the way needed for the things to work.

## Good use cases for this library

* I need to build and deploy an NLP model fast: Following some of the examples should be easy to build simple classifiers, regressors, encoders, siamese architectures, etc.
* I want to test an hypothesis/architecture/representation: For example, let's assume that you have a classifier built with word embeddings and an RNN, you think that there are a lot of misspelled words so you want a digest of char embeddings for each word to learn unknown words. You can easily build this with the library without concerns about preprocessing. Testing word embeddings vs word embeddings + char embeddings you can test your hypothesis fast inside the library and if it is true modify your vanilla keras model the way you like.
* I want a lightweight model: Many transfer learning applications of NLP are huge models and slow on inference time. Our text representations allow building simple architectures with lightweight models with similar costs to a transfer learning application.
* I want a complex architecture: It may be confusing to build a siamese neural network with 3 text representations in each encoder. Coding the generator for those texts is hard and bug-prone, with the library you will only have to chain a few objects.

## Not so good use cases

* I want to make a state of the art model: We don't have state of the art text representations yet.
* I want to interact with the keras model: If you need to interact with the keras model (for example for debugging or interpretability) any part of the model encapsulated by the library will make that task very hard, since we need to prepare the code to stack arbitrary architectures the internal keras model we build has a much more confusing graph that the one you would build using just keras.
* I want to handle multiple outputs and losses (for example for adversarial training): you may struggle connecting models with multiple outputs (more than one output tensor per model) with our library since it's not fully tested/implemented.
* I prefer pytorch

## Installation

```bash
$ pip install GiaNLP
```