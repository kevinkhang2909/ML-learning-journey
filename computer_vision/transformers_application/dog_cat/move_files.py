from pathlib import Path
from tqdm import tqdm


path_train = Path.home() / 'Desktop/dogs-vs-cats/train'
for img in tqdm([*path_train.glob('*.jpg')]):
    category = img.stem.split(".")[0]
    new_folder = path_train / category
    new_folder.mkdir(parents=True, exist_ok=True)
    img.rename(new_folder / img.name)
