# Predicting Accidents in the Mining Industry with a Multi-Task Learning Approach

> A multi-task learning approach to train the mining accident risk prediction models.

![Python 3.x](https://img.shields.io/badge/python-3.x-green.svg)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Inria-Chile/mining-risk-multitask-model)
![CI](https://github.com/Inria-Chile/risotto/workflows/CI/badge.svg)
[![Inria](https://img.shields.io/badge/Made%20in-Inria-%23e63312)](http://inria.cl)
[![License: CeCILLv2.1](https://img.shields.io/badge/license-CeCILL--v2.1-orange)](https://cecill.info/licences.en.html)

This repository contains the source files that support the paper:

* Rodolfo Palma, Luis Martí and Nayat Sánchez-Pi (2020) *Semantic Model of Mining Inspection: A Domain Ontology*. submitted to the [Thirty-Third Annual Conference on Innovative Applications of Artificial Intelligence (IAAI-21)](https://aaai.org/Conferences/AAAI-21/iaai-21-call/).

## Abstract

The mining sector is a very relevant part of the Chilean economy, representing more than 14% of the country’s GDP and more than 50% of its exports. However, mining is also a high-risk activity where health, safety, and environmental aspects are fundamental concerns to take into account to render it viable in the longer term. The Chilean National Geology and Mining Service (SERNAGEOMIN, after its name in Spanish) is in charge of ensuring the safe operation of mines. On-site inspections are their main tool in order to detect issues, propose corrective measures, and track the compliance of those measures.  Consequently, it is necessary to create inspection programs relying on a data-based decision-making strategy.

This paper reports the work carried out in one of the most relevant dimensions of said strategy: predicting the mining worksites accident risk. That is, how likely it is a mining worksite to have accidents in the future. This risk is then used to create a priority ranking that is used to devise the inspection program. Estimating this risk at the government regulator level is particularly challenging as there is a very limited and biased data.

Our main contribution is to apply a multi-task learning approach to train the risk prediction model in such a way that is able to overcome the constraints of the limited availability of data by fusing different sources. As part of this work, we also implemented a human-experience-based model that captures the procedures currently used by the current experts in charge of elaborating the inspection priority ranking.

The mining worksites risk rankings built by model achieve a 121.2% NDCG performance improvement over the rankings based on the currently used experts’ model and outperforms the non-multi-task learning alternatives.

## Installing

To install the project dependencies run:

```zsh
pip install -r requirements.txt
```

## Citing

```bibtex
@techreport{palma2020:mining-accident-risk,
    author = {Palma, Rodolfo and Mart{\'{\i}}, Luis and Sanchez-Pi, Nayat}
    title = {Predicting Accidents in the Mining Industry with a Multi-Task Learning Approach},
    year = {2020},
    institution = {Inria Research Center in Chile},
    note = {submitted to the Thirty-Third Annual Conference on Innovative Applications of Artificial Intelligence (IAAI-21)}
}
```
