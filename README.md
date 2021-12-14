# README #

This README would normally document whatever steps are necessary to get your application up and running.

### What is this repository for? ###

* Quick summary
* Version
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### How do I get the data? ###

* The <query,document> judgements dataset used in this paper is /MSLR-WEB30k/Fold1/train.txt
* To download the data/MSLR-WEB30k/Fold1 please refer to: https://www.microsoft.com/en-us/research/project/mslr
* The dataset is passed in the experiment as 'dataset_path' parameter


* The long tail distribution is extracted for real-world query logs and stored here: data/long_tail/query_click_user.json
* To reproduce the experiments in this paper please use the provided long tail distribution and pass it as 'long_tail_dataset_path' parameter
* if you want to run the experiments with a custom long tail distribution, feel free to use the same json structure(Apache Solr JSON facets response)

### Contribution guidelines ###

* Writing tests
* Code review
* Other guidelines

### Who do I talk to? ###

* Repo owner or admin
* Other community or team contact