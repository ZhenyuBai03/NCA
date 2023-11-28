# NCA
![Alt Text](./SAMPLE/Hi.gif)
Hi! Welcome to NCA. 
## Train:
```bash
python3 train.py --emoji [str] --to_data_path [bool]
```
* `--emoji`: emoji to train on
* `--to_data_path`: whether to save the batch image and pool image to "/data/train"

**Example:**
```bash
python3 train.py --emoji "😘" --to_data_path True
```

## Test: 
```bash
python3 test.py --gen_pool [bool] --weight_path [str]
```
* `--gen_pool`: whether to generate the pool gif(takes time)
* `--weight_path`: path to the weight file in "/data/"


