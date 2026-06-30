from pathlib import Path
from shutil import copy2

import cv2
    
class PhotoModele :
    
    def __init__(self, images : list[Path], destination : Path | None = None):
        self.setImages(images)
        self.name_dos_racine_relative = Path("masques")
        self.name_dos_masques_temp = Path(".temp_masques")
        self.name_dos_masques_accepter = Path("masques")
        self.name_dos_photo_refuser = Path("sans_masque")
        self.destination = destination
        if self.destination is not None:
            if self.destination.is_dir == False:
                raise ValueError("La destination n'est pas un dossier")
            
        self.shape : tuple = (256+128,512+256)


    def setImages(self, newImages : list[Path]):
        newImages = [p for p in newImages if not p.name.endswith("_mask.png")]
        self.images = sorted(newImages)
        self.index = -1
    
    def setDestination(self, newDestiation : Path):
        
        if newDestiation.is_dir == False:
                raise ValueError("La destination n'est pas un dossier")
        self.destination = newDestiation

    def __repr__(self) -> str:
        return self.images[self.index].name
    
    def __str__(self) -> str:
        return self.images[self.index].name
    
    def __bool__(self):
        return self.images != [] and self.images is not None

    
    def getActuel(self):
        img_path = self.images[self.index]
        print(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            raise ValueError(f"Impossible de charger {img_path}")
        self.shape = img.shape
        
        img_rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
        
        mask_path = img_path.parent / self.name_dos_racine_relative / self.name_dos_masques_temp / Path(img_path.name.split('.')[0]+"_mask.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return img_rgb, None

        return img_rgb, mask
    
    def __next__(self):
        self.index += 1
        self.index = self.index % len(self.images)

        return self.getActuel()
    
    def accepter(self, masque):
        img_path = self.images[self.index]
    
        # dossier temp_masques au même niveau que l'image
        if self.destination is None:
            out_dir = img_path.parent / self.name_dos_racine_relative
        else:
            print(f" destination : {self.destination}")
            out_dir = self.destination
        out_dir.mkdir(exist_ok=True)
        out_dir = out_dir / self.name_dos_masques_accepter
        out_dir.mkdir(exist_ok=True)
        # nom fichier
        out_path = out_dir / f"{img_path.stem}_mask.png"
        print(f"enregistrer dans {out_path}")
        
        
        cv2.imwrite(str(out_path), masque.astype("uint8"))
        self.nettoyer()
        return self.getActuel()

    def refuser(self):
        img_path = self.images[self.index]
    
        # dossier temp_masques au même niveau que l'image
        if self.destination is None:
            out_dir = img_path.parent / self.name_dos_racine_relative
        else:
            out_dir = self.destination
        out_dir.mkdir(exist_ok=True)    
        out_dir = out_dir / self.name_dos_photo_refuser
        out_dir.mkdir(exist_ok=True)
        # nom fichier
        out_path = out_dir / f"{img_path.name}"
        print(f"enregistrer dans {out_path}")
        copy2(img_path, out_path)
        self.nettoyer()
        return self.getActuel()

    def nettoyer(self):
        img_path = self.images[self.index]
        # chemin du masque temporaire
        mask_path = (
            img_path.parent
            / self.name_dos_racine_relative
            / self.name_dos_masques_temp
            / f"{img_path.stem}_mask.png"
        )

        # suppression du masque temporaire
        if mask_path.exists():
            mask_path.unlink()
            # dossier parent du masque
            folder = mask_path.parent

            # supprimer le dossier s'il est vide
            if folder.exists() and not any(folder.iterdir()):
                folder.rmdir()
            

        # suppression de l'image de la liste des images à traiter
        self.images.pop(self.index)

        # ajustement de l'index
        if self.images:
            self.index %= len(self.images)
        else:
            raise StopIteration("Toutes les images ont été traitées")
