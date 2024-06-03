# Mitochondrial morphology analysis using confocal images
Author of the original implementation: **Huai-Ching Hsieh**
- Algorithm design and implementation

Modified by: Yu-Te Lin
- Code structure modification

## Usages

### Search the best hypararameters for masking <br>

-n=how many value will be used to build the hyperparameter grid.<br>-r=how many hyperparameter sets will be tested.
<br>(EST: ~1 min per hyperparameter)
```bash
python -m src.img_analysis -m search_mask -i PATH_TO_DATA_FOLDER -t CZI_FILES_TO_TEST -n 3 -r 10
```
Then, you can pick the best hyperparameter by changing the is_best value in the hparam_tested.csv from 'X' to 'V', and save the file.

To create a new hyperparameter file, use this command:

-o, --output: path to the folder for saving the files<br>
-u, --use_nucleus: if specified, save as nucleus.json. Otherwise the file will be saved as tmrm.json
```bash
# next, use the hparam_tested.csv file to create a new hparam.json

python -m src.img_analysis -m sel_best_param -i PATH_TO_THE_CSV -o PATH_TO_OUTPUT

# For instance, this will create a tmrm.json file in new_hparams:
python -m src.img_analysis -m sel_best_param -i ./data/A2/tmrm_masks/hparam_tested.csv -o ./new_hparams
```


### Population mitochondrial analysis<br>
-b (optional): folder storing hyperparameter files (only tmrm.json and nucleus.json are acceptable)
```bash
python -m src.img_analysis -m population -i PATH_TO_DATA_FOLDER -b ./new_hparams
```

### single cell mitochondrial analysis
```bash
python -m src.img_analysis -m sc -i PATH_TO_DATA_FOLDER -e EXPERIMENT_CONDITIONS -d DISH_NAMES -f FRAME_NAMES
```