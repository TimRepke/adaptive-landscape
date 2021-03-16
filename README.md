# Robust Visualisation of Dynamic Text Collections: Measuring and Comparing Dimensionality Reduction Algorithms

This repository contains the code used in our [CHIIR 2021](https://acm-chiir.github.io/chiir2021) publication.  
This work is part of the [Minír Project](https://hpi.de/naumann/projects/web-science/mimir-corpus-exploration-and-knowledge-management.html).

### Abstract
> Visualisations are supposed to provide intuitive ways to explore large document collections. State-of-the-art approaches usually transform high-dimensional representations of documents into 2-dimensional vectors using dimensionality reduction algorithms. These vectors are then placed into a landscape hopefully retaining semantic information regarding similarity from the high-dimensional representation. Traditionally, dimensionality reduction algorithms are developed with static collections in mind. However, many “realworld” document collections, such as news articles, scientific literature, patents, Wikipedia, or tweets, to name a few, grow and evolve over time. Visualising the temporal change of these collections poses various challenges for out-of-the-box dimensionality reduction algorithms.
> In this paper, we propose strategies to adapt existing dimensionality reduction algorithms to incorporate change. These strategies ensure that landscapes at different intervals of the collection are robust with regard to spatio-temporal coherence. Furthermore, we propose metrics to measure the stability over time and

### Source code
* Tested with Python 3.8
* Running the setup.sh should set up all you need in a Python virtual-env within the folder and download relevant datasets.
* `Code` contains all relevant code for the experiments
* `run_models.py` will execute the models and store results in the `TempData` folder.
* `calculate_models.py` / `calculate_metrics.py` will then create plots and metrics.
* `Viewer` contains a small webGL-powered scatterplot viewer

### Reference

> Tim Repke and Ralf Krestel. 2021. Robust Visualisation of Dynamic Text Collections: Measuring and Comparing Dimensionality Reduction Algorithms. In Proceedings of the 2021 Conference on Human Information Interaction and Retrieval (CHIIR '21). Association for Computing Machinery, New York, NY, USA, 255–259. DOI:https://doi.org/10.1145/3406522.3446034

```
@inproceedings{10.1145/3406522.3446034,
author = {Repke, Tim and Krestel, Ralf},
title = {Robust Visualisation of Dynamic Text Collections: Measuring and Comparing Dimensionality Reduction Algorithms},
year = {2021},
isbn = {9781450380553},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3406522.3446034},
doi = {10.1145/3406522.3446034},
booktitle = {Proceedings of the 2021 Conference on Human Information Interaction and Retrieval},
pages = {255–259},
numpages = {5},
keywords = {visual search, dimensionality reduction, corpus exploration},
location = {Canberra ACT, Australia},
series = {CHIIR '21}
}
```
