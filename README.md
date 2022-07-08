![Tests on latest tensorflow version](https://github.com/jian01/GiaNLP/actions/workflows/tests.yml/badge.svg?branch=initial_setup) [![codecov](https://codecov.io/gh/jian01/GiaNLP/branch/initial_setup/graph/badge.svg)](https://codecov.io/gh/jian01/GiaNLP)

# GiaNLP

This is a library developed by me for Mercadolibre and soon to become open source. The library has the goal to be a bridge between NLP preprocessing needed for keras, and custom keras models, allowing the building of complex and huge models with multiple ways of representing text without having to deal with the preprocessing. The library wraps your keras architecture and let you train directly with texts as x values, respecting all the keras API (predict, fit, compile, you can use generators, sequences, multiprocessing, use the model of our library as the input of a vainilla keras model, etc). This is already in use at some limited projects in Mercadolibre.

My release roadmap is:

* Finish little refactors
* Create all the testing and linting pipelines
* Migrate the documentation and examples from Mercadolibre to a more friendly format, along with writing the readme
* Release the first version to pypi (merge [initial_setup](https://github.com/jian01/GiaNLP/tree/initial_setup) to main)

The plan is to finish this one to two weeks from now, meanwhile this is a temporary readme.