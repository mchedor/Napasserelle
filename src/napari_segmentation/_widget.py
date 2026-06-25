from __future__ import annotations

#import json
import re
import shutil
from datetime import datetime
from pathlib import Path
import napari
import numpy as np
import cv2 
from qtpy.QtCore import Qt, QThread, QTimer
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
    QSlider,
)
from qtpy.QtGui import QKeySequence
from napari.utils.colormaps import Colormap

from .Cleaner import CleanerModele  
from .Photo_input import PhotoModele
from typing import Generator
from .detecteur import LoadingWidget
from ._modification_widget import EditWidget


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

        self._model_path: Path | None = None
        self._destination_path: Path | None = None
        self.input_PhotoModele : PhotoModele | None = None
        self.cleaner : CleanerModele = CleanerModele()
        self._image_layer = None
        self._mask_layer = None
        self._mask_shape_layer = None
        self._pred_layer = None
        self._edit_pred_mode : bool = False

        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(400)
        self._spinner_timer.timeout.connect(self._tick_spinner)
        self._spinner_frames = ["Retraining .", "Retraining ..", "Retraining ...", "Retraining"]
        self._spinner_index = 0

        self._model_path_input = QLineEdit()
        self._model_path_input.setPlaceholderText("Path to Unet model (.pth) or model folder")

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._on_browse_model)

        self.loading_widget = LoadingWidget(parent=self)
        self.loading_widget.hide()

        #load_button = QPushButton("Load model")
        #load_button.clicked.connect(self._on_load_model)

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

        

        add_correction_button = QPushButton("Add correction")
        #add_correction_button.clicked.connect(self._on_add_correction)

        self._retrain_button = QPushButton("Retrain")
        #self._retrain_button.clicked.connect(self._on_retrain)

        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("Model"))
        row_model.addWidget(self._model_path_input)
        row_model.addWidget(browse_button)

        row_actions = QHBoxLayout()
        #row_actions.addWidget(load_button)
        row_actions.addWidget(predict_button)
        row_actions.addWidget(self.loading_widget)

        row_input = QHBoxLayout()
        row_input.addWidget(QLabel("Input"))
        row_input.addWidget(self.row_path_input)
        row_input.addWidget(browse_row_button)

        row_destination = QHBoxLayout()
        row_destination.addWidget(QLabel("Destination"))
        row_destination.addWidget(self.destination_path_input)
        row_destination.addWidget(browse_destination_button)


        next_button = QPushButton("Suivant")
        next_button.clicked.connect(self._next_image)

        newt_input = QShortcut(QKeySequence("Right"), self.napari_viewer.window._qt_viewer)
        newt_input.activated.connect(self._next_image)

        accept_button = QPushButton("Accepter")
        accept_button.clicked.connect(self._accept_image)

        refuser_button = QPushButton("Refuser")
        refuser_button.clicked.connect(self._refuser_image)

        
        
        update_button = QPushButton("Update")
        update_button.clicked.connect(self._update_image)

        self.edit_widget = EditWidget(self.cleaner)
        self.edit_widget.modeChanged.connect(self._on_mode_changed)
        self.edit_widget.valuesChanged.connect(self._on_values_changed)


        #row_train = QHBoxLayout()
        #row_train.addWidget(add_correction_button)
        #row_train.addWidget(self._retrain_button)
        #row_train.addStretch(1)
        col_navigation = QVBoxLayout()
        col_navigation.addWidget(next_button)

        row_navigation = QHBoxLayout()
        row_navigation.addWidget(accept_button)
        row_navigation.addWidget(refuser_button)
        #row_navigation.addWidget(reset_button)
        col_navigation.addLayout(row_navigation)


        col_modification = QVBoxLayout()
        col_modification.addWidget(update_button)
        col_modification.addWidget(self.edit_widget)
        



        layout = QVBoxLayout()
        layout.addLayout(row_model)
        layout.addLayout(row_actions)
        layout.addLayout(row_input)
        layout.addLayout(row_destination)
        layout.addLayout(col_navigation)
        layout.addLayout(col_modification)
        #layout.addLayout(row_train)
        layout.addStretch(1)
        self.setLayout(layout)

    def _show_info(self, text: str) -> None:
        QMessageBox.information(self, "Simple CCI Annotator", text)

    def _show_error(self, text: str) -> None:
        QMessageBox.critical(self, "Simple CCI Annotator", text)

    def _accept_image(self):
        if self.input_PhotoModele is not None:
            try:
                img, proba = self.input_PhotoModele.accepter(self._points_to_mask(self._mask_shape_layer, self.input_PhotoModele.shape[:2]))
                self.cleaner.setImageMasque(img, proba)
                self._show_image(img, proba)

            except StopIteration:
                self._show_info("Toute les images ont été traitées")
            except ValueError as e:
                self._show_error(f"impossible de charger {e}")
        else : 
            self._show_error("Il n'y a aucun dossier à traiter")
        
    def _refuser_image(self):
        if self.input_PhotoModele is not None:
            try:
                img, proba = self.input_PhotoModele.refuser()
                self.cleaner.setImageMasque(img, proba)
                self._show_image(img, proba)

            except StopIteration:
                self._show_info("Toute les images ont été traitées")
            except ValueError as e:
                self._show_error(f"impossible de charger {e}")
        else : 
            self._show_error("Il n'y a aucun dossier à traiter")

    def _update_image(self):
        if self.input_PhotoModele is not None:
            try:
                img, proba = self.input_PhotoModele.getActuel()
                self.cleaner.setImageMasque(img, proba)
                self._show_image(img, proba)

            except ValueError as e:
                self._show_error(f"impossible de charger {e}")
        else : 
            self._show_error("Il n'y a aucun dossier à traiter")
    
    def _next_image(self):
        if self.input_PhotoModele is not None:
            try:
                img, proba = next(self.input_PhotoModele)
                self.cleaner.setImageMasque(img, proba)
                self._show_image(img, proba)

            except StopIteration:
                self._show_info("Toute les images ont été traitées")
            except ValueError as e:
                self._show_error(f"impossible de charger {e}")
        else : 
            self._show_error("Il n'y a aucun dossier à traiter")

    def _show_image(self, img, proba):
        if (self._image_layer is None) or (self._image_layer not in self.napari_viewer.layers):
            self._image_layer = self.napari_viewer.add_image(
                img,
                name="Image d'entrée"
            )
        else : 
            self._image_layer.data = img  # type: ignore
        if proba is not None:
            try:
                if (self._pred_layer is None) or (self._pred_layer not in self.napari_viewer.layers):
                    self._pred_layer = self.napari_viewer.add_image(
                        SimpleCciAnnotatorQWidget.resize_back(proba, shape=(img.shape[0], img.shape[1])),
                        name="probability",
                        colormap="magma",
                        opacity=0.5,
                        contrast_limits=(0, 255),
                        blending="additive",
                        visible=False
                    )
                else : 
                    self._pred_layer.data = SimpleCciAnnotatorQWidget.resize_back(proba, shape=(img.shape[0], img.shape[1]))
            except Exception as exc:
                print(exc)
                self._show_error(f"Affichage prediction failed: {exc}")
                return

            try:
                self.cleaner.setImageMasque(img, proba)
                if self._edit_pred_mode:
                    mask_afficher = self.cleaner.getMasqueClean()
                else:
                    polygon = self.cleaner.getMasquePolygon()
            except Exception as exc:
                self._show_error(f"Clean failed: {exc}")
                return
        
            try:
                if self._edit_pred_mode:
                    
                    mask_afficher = SimpleCciAnnotatorQWidget.resize_back(mask_afficher, shape=(img.shape[0],img.shape[1]))
                    if (self._mask_layer is None) or (self._mask_layer not in self.napari_viewer.layers):
                        self._mask_layer = self.napari_viewer.add_labels(
                                mask_afficher,
                                name="mask",
                            )
                        self._mask_layer.color_mode = "direct"
                        self._mask_layer.color = {1: "lime"}
                    else:
                        self._mask_layer.data = mask_afficher
                else:
                    if (self._mask_shape_layer is None) or (self._mask_shape_layer not in self.napari_viewer.layers):
                        self._mask_shape_layer = self.napari_viewer.add_shapes(
                                [p.copy() for p in polygon],
                                shape_type='polygon',
                                face_color='green',
                                name="mask polygon"
                            )
                    else:
                        self._mask_shape_layer.data = [p.copy() for p in polygon]
            except Exception as exc:
                self._show_error(f"affichage failed: {exc}")
                return


    def _points_to_mask(self, layer, shape : tuple[int, int]):
        """
        layer : napari Points layer
        shape : (H, W)
        """
        if len(shape) != 2:
            raise RuntimeError("le masque n'est pas à deux dimension")
        mask = np.zeros(shape, dtype=np.uint8)

        # napari Shapes = liste de polygones
        for poly in layer.data:

            poly = np.asarray(poly)

            # sécurité : enlever polygones invalides
            if poly.shape[0] < 3:
                continue

            # OpenCV attend (x, y)
            pts = poly[:, ::-1].astype(np.int32)

            # dessine le polygone plein
            cv2.fillPoly(mask, [pts], 255)
        mask = (mask > 0).astype(np.uint8) * 255
        print(f"shape du mask : {mask.shape}, max = {np.max(mask)}, min = {np.min(mask)}")
        return mask
    

    def _on_browse_model(self) -> None:
        model_dir, _ = QFileDialog.getOpenFileName(
            self,
            "Select model folder (.pth will be loaded or yolov8n.pth will be copied)"
        )
        if model_dir:
            self._model_path_input.setText(model_dir)
            self._on_load_model()

    def _on_browse_destination(self) -> None:
        destination_dir = QFileDialog.getExistingDirectory(self, "Select destination for retrained model")
        if destination_dir:
            self.destination_path_input.setText(destination_dir)
            try:
                if self.input_PhotoModele is not None:
                    self.input_PhotoModele.setDestination(Path(self.destination_path_input.text()))
            except ValueError as e:
                self._show_error(f"Erreur de destination : {e}")

    def _on_browse_input_row(self) -> None:
        path_row_dir = QFileDialog.getExistingDirectory(self, "Select destination for retrained model")
        if path_row_dir:
            self.row_path_input.setText(path_row_dir)
            extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
            files = [f for f in Path(path_row_dir).iterdir() if f.suffix.lower() in extensions]
            if self.destination_path_input.text():
                self.input_PhotoModele = PhotoModele(files, Path(self.destination_path_input.text()))
            else:
                self.input_PhotoModele = PhotoModele(files)
            self._next_image()

    def _on_mode_changed(self, edit_pred_mode: bool):
        self._edit_pred_mode = edit_pred_mode

        # masque polygon supprimer pour l'edit

        if edit_pred_mode:
            if self._mask_shape_layer is not None:
                if self._mask_shape_layer in self.napari_viewer.layers:
                    self.napari_viewer.layers.remove(self._mask_shape_layer)
        else:
            if self._mask_layer is not None:
                if self._mask_layer in self.napari_viewer.layers:
                    self.napari_viewer.layers.remove(self._mask_layer)
        self._update_image()
            

    def _on_values_changed(self, thresh: int, simplification: float):
        # update cleaner
        self.cleaner.setThresh(thresh)
        self.cleaner.setForce_simplificication(simplification)

        # refresh image
        if self.input_PhotoModele is None:
            return

        try:
            img, proba = self.input_PhotoModele.getActuel()
            self._show_image(img, proba)

        except Exception as e:
            self._show_error(f"Update failed: {e}")

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
        else:
            self._show_error("Select a valid .pth model file or a folder.")
            return

        try:
            #self._spinner_timer.start()
            self.loading_widget.create_model(model_path, (256+128,512+256))
            self._model_path = model_path
            
        except Exception as exc:  # pragma: no cover - GUI runtime guard
            self._show_error(f"Could not load model: {exc}")
            #self._on_retrain_error(f"Could not load model: {exc}")
            return

        if copied_default_model:
            self._show_info(f"No .pth model was found in the folder. Copied bundled model to: {model_path}")

        #self._on_retrain_done(f"Model loaded: {model_path.name}")
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

    def _on_predict(self):
        #image_paths = self.row_path_input.text()
        if self.input_PhotoModele is not None:
            image_paths = self.input_PhotoModele.images

            self.loading_widget.show()
            try:
                self.loading_widget.start(image_paths)
                self.loading_widget.finished.connect(self._on_loading_finished)
            except AttributeError as e:
                self._show_error(f"Erreur de prediction : {e}")
                self.loading_widget.hide()
            except Exception as e:
                self._show_error(f"Erreur de prediction : {e}")
                self._on_loading_finished()            
        else:
            self._show_error("Pas de Photo disponible")
        
    def _on_loading_finished(self):
        self.loading_widget.hide()
        img, proba = self.input_PhotoModele.getActuel()
        #self.cleaner.setImageMasque(img, proba)
        self._show_image(img, proba)    

    def _find_shapes_layer(self):
        layer = self._get_layer_by_name(self.PRED_LAYER_NAME)
        if self._is_shapes_layer(layer):
            return layer

        selected = self.napari_viewer.layers.selection.active
        if self._is_shapes_layer(selected):
            return selected
        return None

    def _tick_spinner(self) -> None:
        self._retrain_button.setText(self._spinner_frames[self._spinner_index % len(self._spinner_frames)])
        self._spinner_index += 1


