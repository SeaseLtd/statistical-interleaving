# README #
To replicate this work just install the requirements.txt in your python environment.

e.g.

using pip
```
pip install -r requirements.txt
```

using Conda
```
conda create --name <env_name> --file requirements.txt
```
### Datasets ###

* The <query,document> judgements dataset used in this paper is /MSLR-WEB30k/Fold1/train.txt
* To download the data/MSLR-WEB30k/Fold1 please refer to: https://www.microsoft.com/en-us/research/project/mslr
* The dataset is passed in the experiment as 'dataset_path' parameter


* The long tail distribution is extracted for real-world query logs and stored here: data/long_tail/query_click_user.json
* To reproduce the experiments in this paper please use the provided long tail distribution and pass it as 'long_tail_dataset_path' parameter
* if you want to run the experiments with a custom long tail distribution, feel free to use the same json structure(Apache Solr JSON facets response)

### Runs ###
Run 9 and 10 have been executed on  Amazon Ec2 machines Z1d 2Xlarge.
All other runs have been executed on Amazon Ec2 machines T2 Xlarge.
The detailed output of each run is in the runs_output folder.

1:
```
python3 -u ./main.py -p ./train.txt -s 234 -q 1000 -m 136 -e 1 -c 10 > runs_output/run-1.txt
```

2:
```
python3 -u ./main.py -p data/MSLR-WEB30k/Fold1/train.txt -s 347 -q 1000 -m 136 -e 1 > runs_output/run-2.txt
```

3-4:
```
python3 -u ./main.py -p data/MSLR-WEB30k/Fold1/train.txt -s 333 -q 1000 -m 136 -e 1 -n 10 -c 10 > runs_output/run-3-4.txt
```

5:
```
python3 -u ./main.py -p data/MSLR-WEB30k/Fold1/train.txt -s 173 -q 100 -m 136 -e 1 -n 10 -c 10 > runs_output/run-5.txt
```

6:
```
python3 -u ./main.py -p data/MSLR-WEB30k/Fold1/train.txt -s 193 -q 100 -m 136 -e 1 -n 10 -c 10 -u 10 > runs_output/run-6.txt
```

7:
```
python3 -u ./main.py -p data/MSLR-WEB30k/Fold1/train.txt -s 777 -q 100 -m 136 -e 1 -c 10 -u 10 > runs_output/run-7.txt
```

8:
```
python3 -u ./main.py -p data/MSLR-WEB30k/Fold1/train.txt -s 33 -q 1000 -m 136 -e 1_long_tail -a data/long-tail-1/query_click_user.json -u 0.02 -n 10 -c 10 > runs_output/run-8.txt
```

9:
```
python3 -u ./main.py -p data/MSLR-WEB30k/Fold1/train.txt -s 337 -q 1000 -m 70 -e 1_long_tail -a data/long-tail-1/query_click_user.json -u 0.125 -n 10 -c 10 > runs_output/run-9.txt
```

10:
```
python3 -u ./main.py -p data/MSLR-WEB30k/Fold1/train.txt -s 571 -q 1000 -m 70 -e 1_long_tail -a data/long-tail-1/query_click_user.json -u 0.250 -n 10 -c 10 > runs_output/run-10.txt
```

11:
```
python3 -u ./main.py -p data/MSLR-WEB30k/Fold1/train.txt -s 133 -q 1000 -m 136 -e 1_long_tail -a data/long-tail-1/query_click_user.json -u 0.02 -c 10 > runs_output/run-11.txt
```

12:
```
python3 -u ./main.py -p data/MSLR-WEB30k/Fold1/train.txt -s 189 -q 1000 -m 136 -e 1_long_tail -a data/long-tail-1/query_click_user.json -u 0.02 -n 10 -c 10 -r True > runs_output/run-12.txt
```

13
```
python3 -u ./main.py -p data/MSLR-WEB30k/Fold1/train.txt -s 133 -q 1000 -m 136 -e 1_long_tail -a data/long-tail-1/query_click_user.json -u 0.02 -c 10 -r True > runs_output/run-13.txt
```