from pathlib import Path
from shutil import copy2
import warnings

import cv2

class PhotoInput :
    
    def __init__(self, images : list[Path]):
        self.setImages(images)


    def setImages(self, newImages : list[Path]):
        #--------sans pytorch
        newImages = [p for p in newImages if not p.name.endswith("_mask.png")]
        #-----------
        self.images_path = sorted(newImages)
        self.index = -1

    def __repr__(self) -> str:
        return self.images_path[self.index].name
    
    def __str__(self) -> str:
        return self.images_path[self.index].name
    
    def __len__(self):
        return len(self.images_path)

    def __iter__(self):
        for img_path in self.images_path:
            #print(f"img_path : {img_path}")
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

            if img is None:
                raise ValueError(f"Impossible de charger {img_path}")

            # Cas PNG RGBA
            if len(img.shape) == 3 and img.shape[2] == 4:
                img_rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            yield img_rgb

