# Mitochondrial morphology analysis using confocal images
Author of the original implementation: **Huai-Ching Hsieh**
- Algorithm design and implementation

Modified by: Yu-Te Lin
- Code structure modification

## Usages

population mitochondrial analysis
```bash
python -m src.img_analysis -m population -i PATH_TO_DATA_FOLDER
```

population mitochondrial analysis (predefined masks)
```bash
python -m src.img_analysis -m tmrm -i PATH_TO_DATA_FOLDER
```

single cell mitochondrial analysis
```bash
python -m src.img_analysis -m sc -i PATH_TO_DATA_FOLDER -e EXPERIMENT_CONDITIONS -d DISH_NAMES -f FRAME_NAMES
```