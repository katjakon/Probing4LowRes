# Probing4LowRes
Exploring probing for low-resource languages.

# Get Universal Dependecy data
Download the data for this project by running:
```
python get_data.py --tsv_file "languages.tsv"
```

# Preprocessing
Perform the necessary preprocessing and sub-sampling by running:

```
python preprocess.py --data_dir <DATA-DIR> --seed <SEED> --sample_size 500 --pretrained google-bert/bert-base-multilingual-cased --out_dir <OUT-DIR>
```

# Run probing experiments
Run a probing experiment with
```
python run_probe_exp.py --preprocessed_dir <PATH> --out_path <OUT-PATH> --property <PROP> --clf_type <CLF-TYPE>
```
property can either be upos, Case, Gender, Tense or Number<br>
clf_type can be either SGD or MLP

# Analysis

Results and plots can be found in `results/` and in the notebook `analysis.ipynb`.

Calculate vocabulary overlap by running:
```
python vocab_overlap.py --data_dir <PREPROCESS_DIR> --out_dir <PATH> --pretrained google-bert/bert-base-multilingual-cased 
```

Calculate proportion of unknown tokens by running:
```
python prop_unk.py --data_dir <PREPROCESS_DIR> --out_dir <PATH> --pretrained google-bert/ --unk_token_id 100
```

Calculate Shannon Evenness Index by running:

```
python evennes.py --data_dir <PREPROCESS_DIR> --out_dir <PATH> --property <PROPERTY>
```
property can either be upos, Case, Gender, Tense or Number<br>