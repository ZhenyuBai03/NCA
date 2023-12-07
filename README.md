# NCA
![Alt Text](./SAMPLE/Hi.gif)
Hi! Welcome to NCA. 
## Train:
```bash
python3 main.py --emoji [str] --to_data_path [bool]
```
* `--emoji`: emoji to train on
* `--to_data_path`: whether to save the batch image and pool images to "/data/train"
> During the Training, the target image will also be saved to ./data.
 
**Example:**
```bash
python3 main.py --emoji "😘" --to_data_path True
```

## Test: 
```bash
python3 test.py --gen_pool [bool] --weight_path [str]
```
* `--gen_pool`: whether to generate the pool gif(takes time)
* `--weight_path`: path to the weight file in "/data/"
  * Currently there are 4 weight files available:
    * "😢": "./data/CA_Model_TEAR.pt"
    * "🙋": "./data/CA_Model_Hi.pt"
    * "🤑": "./data/CA_Model_Money.pt"
    * "😘": "./data/CA_Model_😘.pt"


