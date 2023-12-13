# NCA
![HI.gif](./SAMPLE/Hi.gif)

Hi! Welcome to NCA emoji generation. 

See [Sample Folder](./SAMPLE) for more examples.

## How to Train?
```bash
$ python3 main.py --emoji [str] --to_data_path [bool]
```
* `--emoji`: emoji to train on
* `--to_data_path`: whether to save the batch image and pool images to "/data/train/batch" and "/data/train/pool"

* The batch_img shows the batch training result of each epoch and is ordered by loss. 
* The pool_img shows the whole pool training result of each epoch.
> Before the Training, the target image will also be saved to data/target_img.

> Tensorboard will be opened automatically from the terminal if using MacOS

 
**Example:**
```bash
$ python3 main.py --emoji 😘 --to_data_path True
```

## How to Test? 
The testing process will generate a gif of the emoji based on the weight file you choose.
The gif will be saved to 'SAMPLE/' folder.
```bash
$ python3 test.py --weight_file [str] --pool [bool] --speed [str]
```
* `--weigh_file`: the name of the weight_file in "/data/weights"
* `--pool`: whether to use the pool training result to generate gif for pool visualization
* `--speed`: the speed of the gif generation, default is 5.0 faster (recommended)
  * Currently there are 3 weight files available:
    * "😢": "./data/weights/CA_Model_TEAR.pt"
    * "🙋": "./data/weights/CA_Model_Hi.pt"
    * "😘": "./data/weights/CA_Model_KISS.pt"
 
**Example:**
```bash
$ python3 test.py --weight_file CA_Model_Hi.pt --pool False --speed 5.0
```


