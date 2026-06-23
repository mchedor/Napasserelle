from __future__ import annotations

#import json
import re
import shutil
from datetime import datetime
from pathlib import Path
import napari
import numpy as np
import cv2 
from qtpy.QtCore import QThread, QTimer
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QShortcut,
)
from qtpy.QtGui import QKeySequence
import torch
from .Ia import UnetWorker  
from .Cleaner import CleanerModele  
from .Photo_input import PhotoInput
from typing import Generator

class CCIUnetWrapper:

    def __init__(self, model_name_or_path: str = "resnext50_32x4d_plusplus_4.pth"):
        self.model_name = ""
        self.res = None
        self.device = "cpu"
        self.model = self._create_model(model_name_or_path, self.device)
        

    @staticmethod
    def _create_model(model_name_or_path, device):
        # Defer ultralytics/torch import so package import and pure helper tests
        # do not fail on systems without a working torch runtime.
        try:
            from segmentation_models_pytorch import UnetPlusPlus

        except Exception as exc:  # pragma: no cover - runtime environment guard
            raise RuntimeError(
                "Failed to import segmentation_models_pytorch/torch. Install a compatible CPU build "
                "for this platform to run model inference or training."
            ) from exc
        model = UnetPlusPlus(
            encoder_name="resnext50_32x4d",
            encoder_weights=None,
            in_channels=3,
            classes=1
        ).to(device)

        model.load_state_dict(torch.load(model_name_or_path, map_location=device,weights_only=True))
        model.eval()
        return model

    # @classmethod
    # def load_model_by_name(cls, model_name: str, basedir: str = 'models'):
    #     return cls(yolomodel(None, name=model_name, basedir=basedir), model_name=model_name, basedir=basedir)

    # @classmethod
    # def new_model(cls, config=yolo.models.Config2D, model_name: str = "latest", basedir: str = 'models'):
    #     return cls(yolomodel(config, name=model_name, basedir=basedir), model_name=model_name, basedir=basedir)

    def load_model(self, weights_path: Path):
        self.model = self._create_model(weights_path, self.device)

    def predict(self, img):

        self.model.eval()

        with torch.no_grad():
            
            size = (256+128, 512+256)
            import albumentations as A
            transform = A.Compose([
                A.Resize(size[0], size[1]),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ])

            aug = transform(image=img)
            img_t = aug["image"]

            tensor = torch.from_numpy(img_t).float()
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

            # 5. forward
            out = self.model(tensor/ 255.0)

            # 6. sigmoid (binary segmentation)
            prob = torch.sigmoid(out)

            return prob[0, 0].cpu().numpy()

class SimpleCciAnnotatorQWidget(QWidget):
    """SMP visualisation for napari.

    Workflow:
    1) Select model path (.pth) and load model
    2) Predict masque on the active image
    3) User edits shapes
    4) save
    5) next photo
    """

    PRED_LAYER_NAME = "Masque"

    def __init__(self, napari_viewer: napari.Viewer):
        super().__init__()
        print("Bienvenu dans ce plugin !")
        self.napari_viewer = napari_viewer
        self.setWindowTitle("Simple CCI Annotator")

        self._unet: UnetWorker | None = None
        self._model_path: Path | None = None
        self._destination_path: Path | None = None
        self.input_PhotoInput : PhotoInput | None = None
        self.cleaner : CleanerModele = CleanerModele()
        self._image_iter = None
        self._image_layer = None
        self._mask_layer = None
        self._mask_shape_layer = None
        self._pred_layer = None

        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(400)
        self._spinner_timer.timeout.connect(self._tick_spinner)
        self._spinner_frames = ["Retraining .", "Retraining ..", "Retraining ...", "Retraining"]
        self._spinner_index = 0

        self._model_path_input = QLineEdit()
        self._model_path_input.setPlaceholderText("Path to Unet model (.pth) or model folder")

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._on_browse_model)

        load_button = QPushButton("Load model")
        load_button.clicked.connect(self._on_load_model)

        predict_button = QPushButton("Predict")
        predict_button.clicked.connect(self._on_predict)

        self.row_path_input = QLineEdit()
        self.row_path_input.setPlaceholderText("Path of dir of rows")

        browse_row_button = QPushButton("Browse")
        browse_row_button.clicked.connect(self._on_browse_input_row)

        self.destination_path_input = QLineEdit()
        self.destination_path_input.setPlaceholderText("Path to save mask")

        browse_destination_button = QPushButton("Browse")
        browse_destination_button.clicked.connect(self._on_browse_destination)

        next_button = QPushButton("Suivant")
        next_button.clicked.connect(self._next_image)

        newt_input = QShortcut(QKeySequence("Right"), self.napari_viewer.window._qt_viewer)
        newt_input.activated.connect(self._next_image)

        add_correction_button = QPushButton("Add correction")
        #add_correction_button.clicked.connect(self._on_add_correction)

        self._retrain_button = QPushButton("Retrain")
        #self._retrain_button.clicked.connect(self._on_retrain)

        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("Model"))
        row_model.addWidget(self._model_path_input)
        row_model.addWidget(browse_button)

        row_actions = QHBoxLayout()
        row_actions.addWidget(load_button)
        row_actions.addWidget(predict_button)

        row_input = QHBoxLayout()
        row_input.addWidget(QLabel("Input"))
        row_input.addWidget(self.row_path_input)
        row_input.addWidget(browse_row_button)

        row_destination = QHBoxLayout()
        row_destination.addWidget(QLabel("Destination"))
        row_destination.addWidget(self.destination_path_input)
        row_destination.addWidget(browse_destination_button)

        #row_train = QHBoxLayout()
        #row_train.addWidget(add_correction_button)
        #row_train.addWidget(self._retrain_button)
        #row_train.addStretch(1)
        
        row_navigation = QHBoxLayout()
        row_navigation.addWidget(next_button)


        layout = QVBoxLayout()
        layout.addLayout(row_model)
        layout.addLayout(row_actions)
        layout.addLayout(row_input)
        layout.addLayout(row_destination)
        layout.addLayout(row_navigation)
        #layout.addLayout(row_train)
        layout.addStretch(1)
        self.setLayout(layout)

    def _show_info(self, text: str) -> None:
        QMessageBox.information(self, "Simple CCI Annotator", text)

    def _show_error(self, text: str) -> None:
        QMessageBox.critical(self, "Simple CCI Annotator", text)

    def _next_image(self):
        if self._image_iter is not None:
            try:
                img = next(self._image_iter)
                self._show_image(img)

            except StopIteration:
                self._show_info("Toute les images ont été traitées")
        else : 
            self._show_error("Il n'y a aucun dossier à traiter")

    def _show_image(self, img):
        if (self._image_layer is None) or (self._image_layer not in self.napari_viewer.layers):
            self._image_layer = self.napari_viewer.add_image(
                img,
                name="Image d'entrée"
            )
        else : 
            print("changement image")
            self._image_layer.data = img  # type: ignore


    def _on_browse_model(self) -> None:
        model_dir, _ = QFileDialog.getOpenFileName(
            self,
            "Select model folder (.pth will be loaded or yolov8n.pth will be copied)"
        )
        if model_dir:
            self._model_path_input.setText(model_dir)

    def _on_browse_destination(self) -> None:
        destination_dir = QFileDialog.getExistingDirectory(self, "Select destination for retrained model")
        if destination_dir:
            self.destination_path_input.setText(destination_dir)

    def _on_browse_input_row(self) -> None:
        path_row_dir = QFileDialog.getExistingDirectory(self, "Select destination for retrained model")
        if path_row_dir:
            self.row_path_input.setText(path_row_dir)
            extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
            files = [f for f in Path(path_row_dir).iterdir() if f.suffix.lower() in extensions]
            self.input_PhotoInput = PhotoInput(files)
            self._image_iter = iter(self.input_PhotoInput)
            self._next_image()

    def _on_load_model(self) -> None:
        model_input = self._model_path_input.text().strip()
        if not model_input:
            self._show_error("Model path cannot be empty. Select a .pth file or a folder.")
            return

        model_path_input = Path(model_input)
        if not model_path_input.exists():
            self._show_error("Model path does not exist.")
            return

        model_path: Path
        copied_default_model = False

        if model_path_input.is_file():
            if model_path_input.suffix.lower() != ".pth":
                self._show_error("Select a valid .pth model file or a folder.")
                return
            model_path = model_path_input
        elif model_path_input.is_dir():
            pt_files = sorted(model_path_input.glob("*.pth"))
            if pt_files:
                model_path = pt_files[0]
            else:
                default_model_source = Path(__file__).parent / "models" / "yolov8n.pth"
                if not default_model_source.exists():
                    self._show_error("No .pth found in selected folder, and bundled yolov8n.pth is missing.")
                    return
                self._show_error("No .pth found in selected folder, and bundled yolov8n.pth is missing.")
                return
                #model_path = model_path_input / "yolov8n.pth"
                #shutil.copy2(default_model_source, model_path)
                #copied_default_model = True
        else:
            self._show_error("Select a valid .pth model file or a folder.")
            return

        try:
            self._spinner_timer.start()
            self._unet = UnetWorker(str(model_path))
            self._model_path = model_path
            
        except Exception as exc:  # pragma: no cover - GUI runtime guard
            #self._show_error(f"Could not load model: {exc}")
            self._on_retrain_error(f"Could not load model: {exc}")
            return

        if copied_default_model:
            self._show_info(f"No .pth model was found in the folder. Copied bundled model to: {model_path}")

        self._on_retrain_done(f"Model loaded: {model_path.name}")
        #self._show_info(f"Model loaded: {model_path.name}")

    def _get_active_image_layer(self):
        layer = self.napari_viewer.layers.selection.active
        if layer is None:
            self._show_error("Select an image layer first.")
            return None

        if getattr(layer, "data", None) is None:
            self._show_error("Active layer has no image data.")
            return None
        return layer

    def _is_image_layer(self, layer) -> bool:
        return layer is not None and layer.__class__.__name__.lower().endswith("image")

    def _get_single_image_layer(self):
        image_layers = [layer for layer in self.napari_viewer.layers if self._is_image_layer(layer)]
        if len(image_layers) == 0:
            self._show_error("No image layer found.")
            return None
        if len(image_layers) > 1:
            self._show_error("Multiple image layers found. Keep only one image layer before adding a correction.")
            return None
        return image_layers[0]

    def _get_layer_by_name(self, name: str):
        for layer in self.napari_viewer.layers:
            if getattr(layer, "name", None) == name:
                return layer
        return None

    def _is_shapes_layer(self, layer) -> bool:
        return layer is not None and layer.__class__.__name__.lower().endswith("shapes")

    @staticmethod
    def resize_back(mask, shape, interpolation = cv2.INTER_NEAREST):
        #cv2.INTER_LINEAR
        #cv2.INTER_CUBIC
        return cv2.resize(mask, (shape[1], shape[0]), interpolation=interpolation)



    def _on_predict(self) -> None:
        if self._unet is None:
            self._show_error("Load a model first.")
            return


        if self._image_layer is None:
            return

        image_data = np.asarray(self._image_layer.data) # type: ignore
        if image_data.ndim <= 2:
            self._show_error("Unsupported image shape.")
            return

        try:
            
            #print(image_data.shape)
            prediction = self._unet.run_one_inference(image_data)
        except Exception as exc:  # pragma: no cover - GUI runtime guard
            print(exc)
            self._show_error(f"Prediction failed: {exc}")
            return

        try:
            if (self._pred_layer is None) or (self._pred_layer not in self.napari_viewer.layers):
                self._pred_layer = self.napari_viewer.add_image(
                    SimpleCciAnnotatorQWidget.resize_back(prediction, shape=(image_data.shape[0], image_data.shape[1])),
                    name="probability",
                    colormap="magma",
                    opacity=0.5,
                    contrast_limits=(0, 255),
                    blending="additive",
                    visible=False
                )
            else : 
                self._pred_layer.data = SimpleCciAnnotatorQWidget.resize_back(prediction, shape=(image_data.shape[0], image_data.shape[1]))
        except Exception as exc:
            print(exc)
            self._show_error(f"Affichage prediction failed: {exc}")
            return

        try:
            self.cleaner.setImageMasque(image_data, prediction)
            polygon = self.cleaner.getMasquePolygon()
            mask_afficher = self.cleaner.getMasqueClean()
        except Exception as exc:
            self._show_error(f"Clean failed: {exc}")
            return
    
        try:
            mask_afficher = SimpleCciAnnotatorQWidget.resize_back(mask_afficher, shape=(image_data.shape[0],image_data.shape[1]))
            if (self._mask_layer is None) or (self._mask_layer not in self.napari_viewer.layers):
                self._mask_layer = self.napari_viewer.add_labels(
                        mask_afficher,
                        name="mask"
                    )
            else:
                self._mask_layer.data = mask_afficher

            if (self._mask_shape_layer is None) or (self._mask_shape_layer not in self.napari_viewer.layers):
                self._mask_shape_layer = self.napari_viewer.add_shapes(
                        polygon,
                        shape_type='polygon',
                        face_color='green',
                        name="mask polygon"
                    )
            else:
                self._mask_shape_layer.data = polygon
        except Exception as exc:
            self._show_error(f"affichage failed: {exc}")
            return
         

    def _find_shapes_layer(self):
        layer = self._get_layer_by_name(self.PRED_LAYER_NAME)
        if self._is_shapes_layer(layer):
            return layer

        selected = self.napari_viewer.layers.selection.active
        if self._is_shapes_layer(selected):
            return selected
        return None

    def _to_safe_stem(self, name: str) -> str:
        stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
        return stem or "image"

    def _get_model_root(self) -> Path | None:
        if self._model_path is None:
            return None
        return self._model_path.parent

    def _as_rgb_uint8(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.ndim == 3 and image.shape[-1] > 3:
            image = image[..., :3]

        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image


    """
    def _on_add_correction(self) -> None:
        destination_text = self.destination_path_input.text().strip()
        self._destination_path = Path(destination_text) if destination_text else None
        model_root = self._get_model_root()
        if model_root is None:
            self._show_error("Load a model first.")
            return

        image_layer = self._get_single_image_layer()
        if image_layer is None:
            return

        shapes_layer = self._find_shapes_layer()
        if shapes_layer is None:
            self._show_error(f"No shapes layer found. Use '{self.PRED_LAYER_NAME}' or select a shapes layer.")
            return

        image_data = self._as_rgb_uint8(np.asarray(image_layer.data))
        h, w = image_data.shape[:2]
        if h <= 0 or w <= 0:
            self._show_error("Image has invalid size.")
            return

        corrections_root = model_root / "corrections"
        corrections_root.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{self._to_safe_stem(getattr(image_layer, 'name', 'image'))}_{stamp}"
        image_out = corrections_root / f"{stem}.png"
        label_out = corrections_root / f"{stem}.txt"

        vectors: list[tuple[int, list[tuple[float, float]]]] = []
        for shape in np.asarray(shapes_layer.data, dtype=float):
            pts: list[tuple[float, float]] = []
            for y, x in shape:
                xn = float(np.clip(x / w, 0.0, 1.0))
                yn = float(np.clip(y / h, 0.0, 1.0))
                pts.append((xn, yn))
            if pts:
                vectors.append((0, pts))

        try:
            Image.fromarray(image_data).save(image_out)
            save_vectors_to_txt(vectors, label_out)
            # Use the larger image dimension as the training image size
            img_size = max(h, w)
            self._create_training_config(corrections_root, image_size=img_size)
        except Exception as exc:  # pragma: no cover - GUI runtime guard
            self._show_error(f"Could not save correction: {exc}")
            return

        self._show_info(f"Saved correction to: {corrections_root}")

    def _on_retrain(self) -> None:
        if self._unet is None or self._model_path is None:
            self._show_error("Load a model first.")
            return

        model_root = self._model_path.parent
        corrections_root = model_root / "corrections"

        if not corrections_root.exists():
            self._show_error("No corrections found. Add at least one correction first.")
            return

        self._retrain_button.setEnabled(False)
        self._spinner_index = 0
        self._spinner_timer.start()

        self._retrain_worker = _RetrainWorker(
            yolo=self._unet,
            model_root=model_root,
            destination_path=self._destination_path,
            parent=self,
        )
        self._retrain_worker.finished.connect(self._on_retrain_done)
        self._retrain_worker.failed.connect(self._on_retrain_error)
        self._retrain_worker.start()
    """

    def _tick_spinner(self) -> None:
        self._retrain_button.setText(self._spinner_frames[self._spinner_index % len(self._spinner_frames)])
        self._spinner_index += 1

    def _on_retrain_done(self, message: str) -> None:
        self._spinner_timer.stop()
        self._retrain_button.setText("Retrain")
        self._retrain_button.setEnabled(True)
        self._show_info(message)

    def _on_retrain_error(self, message: str) -> None:
        self._spinner_timer.stop()
        self._retrain_button.setText("Retrain")
        self._retrain_button.setEnabled(True)
        self._show_error(message)
