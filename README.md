# DeepWalk_R

Introduction
===========
This is a repository for [DeepWalk](http://www.perozzi.net/publications/14_kdd_deepwalk.pdf) tutorial with **R**. I used [Karate](https://rdrr.io/cran/igraphdata/man/karate.html) social network  dataset in igraphdata packages. I just followed [SkipGram](https://blogs.rstudio.com/tensorflow/posts/2017-12-22-word-embeddings-with-keras/) model code and applied Karate dataset. 

Requirement
===========
-   Keras
-   tidyverse
-   igraph
-   igraphdata
-   reticulate


Result
===========
Output is almost similar to plot in DeepWalk paper.

Input : Karate Graph

<img width = "450" heigth = "400" src = 
https://user-images.githubusercontent.com/37679460/57570416-abc22d00-743c-11e9-9779-46570b0cadbc.png>


Output : Representation

<img width = "450" heigth = "400" src = 
https://user-images.githubusercontent.com/37679460/57570437-d57b5400-743c-11e9-9fda-82ef260cd277.png>
