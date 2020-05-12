# mctseq_over_under_splitting 

Analysis of over- and under-splitting of clustering for snmC2T-seq datasets.

This repository contains code and examples to evaluate the level of over- and under-splitting of a given clustering of a snmC2T-seq dataset. snmC2T-seq is a multi-modal single-cell sequencing method by [Luo et al.](https://www.biorxiv.org/content/10.1101/2019.12.11.873398v1) to measure transcriptomes, DNA methylomes, and chromatin accessibilities at single nucleus level. Taking advantage of the multi-modal measurements, this analysis aims to resolve some lumpers-and-splitters arbitrariness in the practice of clustering single-cell sequencing datasets. By cross referencing complimentary data modalities of the snmC2T-seq dataset, we developed metrics and visualizations to evaluate whether a given cell cluster is over-split or under-split.

For more information and to cite this work:
- [Luo, C. et al. Single nucleus multi-omics links human cortical cell regulatory genome diversity to disease risk variants. bioRxiv 2019.12.11.873398 (2019) doi:10.1101/2019.12.11.873398](https://www.biorxiv.org/content/10.1101/2019.12.11.873398v1)

Code contributors: [Fangming Xie](mailto:f7xie@ucsd.edu)
Contact: [Eran Mukamel](mailto:emukamel@ucsd.edu)

## Installation
Step 1: Clone this repo.
```bash
git clone https://github.com/FangmingXie/mctseq_over_under_splitting.git
cd mctseq_over_under_splitting 
```

Step 2: Set up a conda environment and install dependent packages. (Skip this step if not needed.)
```bash
conda env create -f environment.yml # create an env named scf_dev
source activate scf_dev
```

Step 3: Download the snmC2T-seq data using [this](https://drive.google.com/u/0/uc?export=download&confirm=FGn2&id=1VnFHluULJJ-hKJzwJFb7TPDpRPgrUw-I) google drive link. (gene-by-cell matrices and metadata)

Step 4: Uncompress the data and put it under the directory of this repository
```bash
# assuming mctseq_over_under_splitting is the current directory
tar -xvf data.tar.gz
```

## Usage
```./over-under-splitting-analysis.ipynb``` contains the main code and explanation of the analysis.

