# Corpus-Show
[![Contributor Covenant](https://img.shields.io/badge/contributor%20covenant-v2.0%20adopted-black.svg)](code_of_conduct.md)
[![Python Version](https://img.shields.io/badge/python-3.6%2C3.7%2C3.8-black.svg)](code_of_conduct.md)
![Pypi Version](https://img.shields.io/pypi/v/corpusshow.svg)
![Code convention](https://img.shields.io/badge/code%20convention-pep8-black)

Corpus-Show[코뿔:소] helps to understand the corpus data distribution through various values generated from Sentence Transformer.
- Corpus-Show performs sentence embedding via Sentence Transformers, a Python framework for state-of-the-art sentence, text and image embeddings. [[Paper]](https://arxiv.org/abs/1908.10084) [[Documenttion]](https://www.sbert.net/) [[Huggingface model]](https://huggingface.co/sentence-transformers/paraphrase-xlm-r-multilingual-v1) 
- You can visualize the embedded sentences of each document generated from SentenceTransformers.
- Corpus-Show can also generate clusters with sentences embedded array through [Scikit-Learn KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html).
- The sentence transformer model is downloaded through the hugging face interface, and the default model is set to 'paraphrase-xlm-r-multilingual-v1', which supports multiple languages. However, you can easily input your custom model as a sentence transformer model through the hugging face interface. It is also easy to fine-tune via SBERT. For more models, please see [this page](https://huggingface.co/sentence-transformers).




<br>

# Installation
  ```cmd
  $ pip install corpusshow
  ```
<br>

# Tutorial
We provide tutorial notebooks for all the features we offer. We plan to provide additional docstrings or documentation from the official distribution version (major version 1 or higher).

1. Main-tutorials: https://github.com/DSDanielPark/corpus-show/blob/main/tutorials/corpusshow_tutorial.ipynb
2. Sub-tutorial-folder: https://github.com/DSDanielPark/corpus-show/blob/main/tutorials
 

<Br>

# Main Feature
It helps to create a simple but useful plot as shown below with a simple dataframe and column names as input, such as the following [BBC sample dataset](`http://mlg.ucd.ie/datasets/bbc.html`) in `./data/bbc_news_dataset.csv`.


|     | news  | topic |
|:---:|:----|:----:|
|0|Oil rebounds from weather effect (...)|business|
|1|Indonesia 'declines debt freeze' (...)|business|
|...|...|...|
|601|EU software patent law faces axe (...)|tech|


![](https://github.com/DSDanielPark/corpus-show/blob/main/tutorials/readme_fig1.png)
![](https://github.com/DSDanielPark/corpus-show/blob/main/tutorials/readme_fig2.png)



# Use Case
[1] [Korean-news-topic-classification-using-KO-BERT](https://github.com/DSDanielPark/fine-tuned-korean-BERT-news-article-classifier): all plots were created through Corpus-Show and Quick-Show.

# References
[1] Scikit-Learn https://scikit-learn.org <br>
[2] Matplotlib https://matplotlib.org/ <br>
[3] Huggingface Sentence Transformer https://huggingface.co/sentence-transformers <Br>
[4] SBERT https://www.sbert.net/

### Contacts
Project Owner(P.O): [Daniel Park, South Korea](https://github.com/DSDanielPark) 
e-mail parkminwoo1991@gmail.com <br>
Maintainers: [Daniel Park, South Korea](https://github.com/DSDanielPark) 
e-mail parkminwoo1991@gmail.com
