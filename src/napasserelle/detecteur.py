from pathlib import Path
import traceback
from torch.types import Tensor
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QApplication
from qtpy.QtCore import Qt, QThread, Signal



from cv2.typing import MatLike



# ---------------------------
# Thread pour simuler une tâche longue
# ---------------------------
class WorkerThread(QThread):
    progress = Signal(int)  # signal pour la progression
    finished = Signal()     # signal quand terminé
    error = Signal(str)     # signal d'erreur

    def __init__(self, model_path : Path, image_paths : list[Path], resolution : tuple[int, int], batch_size : int = 8):
        super().__init__()
        
        self.model_path : Path = model_path
        self.image_paths : list[Path] = image_paths
        self.resolution : tuple[int, int] = resolution
        self.batch_size : int= batch_size

    # -------------------------
    # 1. LOAD MODEL
    # -------------------------
    def load_model(self):
        
        self.model = self.UnetPlusPlus(
            encoder_name="resnext50_32x4d",
            encoder_weights=None,
            in_channels=3,
            classes=1
        ).to(self.device)
        

        self.model.load_state_dict(self.torch.load(self.model_path, map_location=self.device,weights_only=True))
        self.model.eval()

    # -------------------------
    # 2. PREPROCESS SINGLE IMAGE
    # -------------------------
    def preprocess(self, path : Path) -> Tensor:
        image = self.cv2.imread(path)
        if image is None:
            raise ValueError(f"Image invalide: {path}")

        image = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2RGB)

        image = self.transform(image)

        return image

    # -------------------------
    # 3. MAKE BATCH
    # -------------------------
    def make_batch(self, batch_paths : list[Path]) -> tuple[Tensor|None, list[Path]]:
        images = [self.preprocess(p) for p in batch_paths]

        if len(images) == 0:
            return None, []

        return self.torch.stack(images).to(self.device, non_blocking=True), batch_paths

    # -------------------------
    # 4. INFERENCE
    # -------------------------
    def predict(self, batch_tensor):
        with self.torch.no_grad():
            preds = self.model(batch_tensor/255)
            preds = self.torch.sigmoid(preds)
        return preds

    # -------------------------
    # 5. POSTPROCESS SINGLE MASK
    # -------------------------
    def postprocess(self, pred, path : Path):
   
        # dossier .temp_masques au même niveau que l'image
        out_dir = path.parent / "masques"
        out_dir.mkdir(exist_ok=True)
        out_dir = out_dir / ".temp_masques"
        out_dir.mkdir(exist_ok=True)
        # nom fichier
        out_path = out_dir / f"{path.stem}_mask.png"

        # sauvegarde
        pred = self.np.squeeze(pred)
        pred = (pred * 255).astype(self.np.uint8)
        self.cv2.imwrite(str(out_path), pred)

    # -------------------------
    # 6. RUN PIPELINE
    # -------------------------
    def run(self):
        try:
            partie_import = 20
            total_steps = 5  # imports + model + data loop
            step = 0

            # =========================
            # 1. IMPORTS LOURDS (progressif)
            # =========================
            import torch
            step += 1
            self.progress.emit(int((step / total_steps) * partie_import))
            

            import cv2
            step += 1
            self.progress.emit(int((step / total_steps) * partie_import))

            import numpy as np
            step += 1
            self.progress.emit(int((step / total_steps) * partie_import))

            from segmentation_models_pytorch import UnetPlusPlus
            step += 1
            self.progress.emit(int((step / total_steps) * partie_import))

            from torchvision.transforms import v2
            step += 1
            self.progress.emit(int((step / total_steps) * partie_import))

            # expose libs en attributs si besoin ailleurs
            self.torch = torch
            self.cv2 = cv2
            self.np = np
            self.UnetPlusPlus = UnetPlusPlus
            self.v2 = v2

            self.device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize(self.resolution),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ),
            ])

            self.load_model()

            total = len(self.image_paths)

            processed = 0

            for i in range(0, total, self.batch_size):

                batch_paths = self.image_paths[i:min(i + self.batch_size, total)]

                batch_tensor, valid_paths = self.make_batch(batch_paths)

                preds = self.predict(batch_tensor)
                preds = preds.detach().cpu().numpy()

                for pred, path in zip(preds, valid_paths):
                    self.postprocess(pred, path)

                processed += len(batch_paths)
                self.progress.emit(partie_import + int(processed * (100-partie_import) / total))

            self.finished.emit()

        except Exception as e:
            print("CRASH DETECTED")
            print(traceback.format_exc(), flush=True)
            self.error.emit(str(e))


# ---------------------------
# Widget de chargement
# ---------------------------
class LoadingWidget(QWidget):
    progress = Signal(int)
    finished = Signal()  # signal pour dire "on a fini"
    error = Signal(str)     # signal d'erreur

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.model_path : Path | None = None
        self.resolution : tuple[int, int] | None = None
        self.device = "cpu"

        self.label = QLabel("Chargement en cours...")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        layout.addWidget(self.label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

        self.progress.connect(self.progress_bar.setValue)
        #self.finished.connect(self.on_finished)

        self._thread : WorkerThread   # sera créé à chaque start

    def create_model(self, model_path : Path, resolution : tuple[int, int], device : str = "cpu"):
        self.model_path = model_path
        self.resolution = resolution
        self.device = device

    def start(self, image_paths):
        if self.model_path is not None and self.resolution is not None:
            self.progress_bar.setValue(0)

            self._thread = WorkerThread(model_path=self.model_path, image_paths=image_paths, resolution=self.resolution)

            self._thread.progress.connect(self.progress_bar.setValue)
            self._thread.finished.connect(self.on_finished)
            self._thread.error.connect(self.error.emit)

            self._thread.start()
            #self.on_finished()
        else:
            raise AttributeError("Aucun modele n'est disponible")

    def on_finished(self):
        self.finished.emit()  # informe le SceneManager que c'est fini


