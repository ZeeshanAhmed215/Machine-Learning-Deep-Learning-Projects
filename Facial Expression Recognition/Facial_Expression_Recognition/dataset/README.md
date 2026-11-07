# Dataset folder

The **FER-2013** dataset is **not included** (size and licence). Download it yourself
(for example from Kaggle: "FER-2013", folder version `msambare/fer2013`, or the original
`fer2013.csv`) and place it here using ONE of these layouts:

**A) Image folders**

    dataset/
        train/
            angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/
        test/
            angry/  disgust/  fear/  happy/  neutral/  sad/  surprise/

**B) Original CSV**

    dataset/
        fer2013.csv        (columns: emotion, pixels, Usage)

Then check it with:

    python -m src.data_loader

This prints the structure, class counts, corrupt files and duplicates, and saves
`results/class_distribution.png` and `results/sample_images.png`.

To keep the data elsewhere, set the environment variable `FER_DATASET_DIR`
(e.g. `set FER_DATASET_DIR=D:\data\fer2013` in Windows CMD) or pass `--dataset-dir`.
