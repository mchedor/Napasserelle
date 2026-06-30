from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2 
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QShortcut
)
from qtpy.QtGui import QKeySequence


from .Cleaner import CleanerModele  
from .Photo_input import PhotoModele
from .detecteur import LoadingWidget
from ._modification_widget import EditWidget

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari
    from napari.layers import Labels, Shapes, Image
    from napari.utils import DirectLabelColormap


class SegmentationBinaire(QWidget):
    """Binary segmentation tools for napari.

    Workflow:
    1) Select a model path (.pth)
    2) Select an input image folder
    3) Predict masks for the images
    4) The user visualizes, modifies, and annotates the results
    5) Accept, refuse, or pass
    """

    def __init__(self, napari_viewer: napari.Viewer):
        
        #~~~~~~~~~~~~~~~~~~ Init
        super().__init__()
        self.napari_viewer = napari_viewer
        self.setWindowTitle("Segmentation Binaire Visualiseur")
        
        #~~~~~~~~~~~~~~~~~~ Path et Object de traitement
        #Model de deeplearning contenu dans self._loading_widget
        self._input_PhotoModele : PhotoModele | None = None
        self._destination_path: Path | None = None
        self._cleaner : CleanerModele = CleanerModele()
        
        #~~~~~~~~~~~~~~~~~~ Layer
        self._mask_shape_layer : Shapes | None = None
        self._mask_layer : Labels | None = None
        self._pred_layer : Image | None = None
        self._image_layer : Image | None = None

        #~~~~~~~~~~~~~~~~~~ Modes
        self._edit_pred_mode : bool = False
        """True si mode édition (Label) et False si mode Shape
        """



        #------------------ Model
        self._model_path_input = QLineEdit()
        self._model_path_input.setPlaceholderText("Path to Unet model (.pth)")

        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._on_browse_model)

        #------------------ Prediction
        predict_button = QPushButton("Predict")
        predict_button.clicked.connect(self._on_predict)

        self._loading_widget = LoadingWidget(parent=self)
        self._loading_widget.hide()

        #------------------ Image
        self._image_folder_path_input = QLineEdit()
        self._image_folder_path_input.setPlaceholderText("Path of dir of image")

        browse_image_folder_button = QPushButton("Browse")
        browse_image_folder_button.clicked.connect(self._on_browse_image_folder_input_)

        #------------------ Destination
        self.destination_path_input = QLineEdit()
        self.destination_path_input.setPlaceholderText("Path to save mask")

        browse_destination_button = QPushButton("Browse")
        browse_destination_button.clicked.connect(self._on_browse_destination)

        #------------------ Navigation
        next_button = QPushButton("Suivant")
        next_button.clicked.connect(self._next_image)

        newt_input = QShortcut(QKeySequence("Right"), self.napari_viewer.window._qt_viewer)
        newt_input.activated.connect(self._next_image)

        accept_button = QPushButton("Accepter")
        accept_button.clicked.connect(self._accept_image)

        refuser_button = QPushButton("Refuser")
        refuser_button.clicked.connect(self._refuser_image)

        #------------------ Update
        update_button = QPushButton("Update")
        update_button.clicked.connect(self._update_image)

        #------------------ Edition Prediction
        self.edit_widget = EditWidget(self._cleaner)
        self.edit_widget.modeChanged.connect(self._on_mode_changed)
        self.edit_widget.valuesChanged.connect(self._on_values_changed)



        #++++++++++++++++++ Model
        row_model = QHBoxLayout()
        row_model.addWidget(QLabel("Model"))
        row_model.addWidget(self._model_path_input)
        row_model.addWidget(browse_button)

        #++++++++++++++++++ Prediction
        row_prediction = QHBoxLayout()
        row_prediction.addWidget(predict_button)
        row_prediction.addWidget(self._loading_widget)

        #++++++++++++++++++ Image
        row_input = QHBoxLayout()
        row_input.addWidget(QLabel("Input"))
        row_input.addWidget(self._image_folder_path_input)
        row_input.addWidget(browse_image_folder_button)

        #++++++++++++++++++ Destination
        row_destination = QHBoxLayout()
        row_destination.addWidget(QLabel("Destination"))
        row_destination.addWidget(self.destination_path_input)
        row_destination.addWidget(browse_destination_button)

        #++++++++++++++++++ Navigation
        col_navigation = QVBoxLayout()
        col_navigation.addWidget(next_button)

        row_navigation = QHBoxLayout()
        row_navigation.addWidget(accept_button)
        row_navigation.addWidget(refuser_button)
        col_navigation.addLayout(row_navigation)

        #++++++++++++++++++ Update + Edition Prediction
        col_modification = QVBoxLayout()
        col_modification.addWidget(update_button)
        col_modification.addWidget(self.edit_widget)
        


        ################## All
        layout = QVBoxLayout()
        layout.addLayout(row_model)
        layout.addLayout(row_prediction)
        layout.addLayout(row_input)
        layout.addLayout(row_destination)
        layout.addLayout(col_navigation)
        layout.addLayout(col_modification)
        layout.addStretch(1)
        self.setLayout(layout)

    def _show_info(self, text: str) -> None:
        QMessageBox.information(self, "Simple CCI Annotator", text)

    def _show_error(self, text: str) -> None:
        QMessageBox.critical(self, "Simple CCI Annotator", text)

    def _next_image(self):
        if self._input_PhotoModele is not None:
            if self._edit_pred_mode: 
                self._show_error(f"Veuillez appliquer les modifications")
            else:
                try:
                    img, proba = next(self._input_PhotoModele)
                    self._cleaner.setMasqueResImg(proba, img)
                    self._show_image(img, proba)

                except StopIteration:
                    self._show_info("Toute les images ont été traitées")
                except ValueError as e:
                    self._show_error(f"impossible de charger {e}")
        else : 
            self._show_error("Il n'y a aucun dossier à traiter")

    def _accept_image(self):
        if self._input_PhotoModele is not None:
            if self._edit_pred_mode: 
                self._show_error(f"Veuillez appliquer les modifications")
            else:
                try:
                    if self._mask_shape_layer is not None:
                        masque_a_stocker = ((self._mask_shape_layer.to_labels(labels_shape = self._input_PhotoModele.shape[:2])>0)*255).astype(np.uint8)

                        img, proba = self._input_PhotoModele.accepter(masque_a_stocker)
                        self._cleaner.setMasqueResImg(proba, img)
                        self._show_image(img, proba)
                    else:
                        self._show_error(f"Il n’y a pas de masque en cours de traitement")
                except StopIteration:
                    self._show_info("Toute les images ont été traitées")
                except ValueError as e:
                    self._show_error(f"impossible de charger {e}")
        else : 
            self._show_error("Il n'y a aucun dossier à traiter")
        
    def _refuser_image(self):
        if self._input_PhotoModele is not None:
            if self._edit_pred_mode: 
                self._show_error(f"Veuillez appliquer les modifications")
            else:
                try:
                    img, proba = self._input_PhotoModele.refuser()
                    self._cleaner.setMasqueResImg(proba, img)
                    self._show_image(img, proba)

                except StopIteration:
                    self._show_info("Toute les images ont été traitées")
                except ValueError as e:
                    self._show_error(f"impossible de charger {e}")
        else : 
            self._show_error("Il n'y a aucun dossier à traiter")

    def _update_image(self):
        if self._input_PhotoModele is not None:
            try:
                img, proba = self._input_PhotoModele.getActuel()
                self._cleaner.setMasqueResImg(proba, img)
                self._show_image(img, proba)

            except ValueError as e:
                self._show_error(f"impossible de charger {e}")
        else : 
            self._show_error("Il n'y a aucun dossier à traiter")
    

    def _show_image(self, img, proba):
        #++++++++++++++++++ Image
        if (self._image_layer is None) or (self._image_layer not in self.napari_viewer.layers):
            image_layer = self.napari_viewer.add_image(
                img,
                name="Image d'entrée"
            )
            if isinstance(image_layer, list):
                self._show_error(f"Affichage de l'image failed: il y a plusieur layer de créer")
            else:
                self._image_layer = image_layer
        else : 
            self._image_layer.data = img 
        
        #++++++++++++++++++ Pour le masquage
        if proba is not None:
            
            #~~~~~~~~~~~~~~~~~~ Prediction
            try:
                if (self._pred_layer is None) or (self._pred_layer not in self.napari_viewer.layers):
                    pred_layer = self.napari_viewer.add_image(
                        SegmentationBinaire.resize_back(proba, shape=(img.shape[0], img.shape[1])),
                        name="probability",
                        colormap="magma",
                        opacity=0.5,
                        contrast_limits=(0, 255),
                        blending="additive",
                        visible=False
                    )
                    if isinstance(pred_layer, list):
                        self._show_error(f"Affichage de la prediction failed: il y a plusieur layer de créer")
                    else:
                        self._pred_layer = pred_layer
                else : 
                    self._pred_layer.data = SegmentationBinaire.resize_back(proba, shape=(img.shape[0], img.shape[1]))
            except Exception as exc:
                print(exc)
                self._show_error(f"Affichage prediction failed: {exc}")
                return


            #~~~~~~~~~~~~~~~~~~ Label
            if self._edit_pred_mode:

                #------------------ Clean
                try:
                    self._cleaner.setMasqueResImg(proba, img)
                    mask_afficher = self._cleaner.getMasqueClean()
                except Exception as exc:
                    self._show_error(f"Clean failed: {exc}")
                    return
                
                #------------------ Masque
                try:
                    mask_afficher = SegmentationBinaire.resize_back(mask_afficher, shape=(img.shape[0],img.shape[1]))
                    if (self._mask_layer is None) or (self._mask_layer not in self.napari_viewer.layers):
                        colormap = DirectLabelColormap()
                        colormap.color_dict[1] = np.array([0, 1, 0, 1])
                        colormap.color_dict[0] = np.array([0, 0, 0, 0])
                        self._mask_layer = self.napari_viewer.add_labels(
                                mask_afficher,
                                name="mask",
                                colormap = colormap
                            )
                    else:
                        self._mask_layer.data = mask_afficher
                except Exception as exc:
                    self._show_error(f"affichage failed: {exc}")
                    return
            
            #~~~~~~~~~~~~~~~~~~ Polygone
            else:

                #------------------ Clean
                try:
                    self._cleaner.setMasqueResImg(proba, img)
                    polygon = self._cleaner.getMasquePolygon()
                except Exception as exc:
                    self._show_error(f"Clean failed: {exc}")
                    return
            
                #------------------ Masque
                try:
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

    def _on_browse_model(self) -> None:
        model_dir, _ = QFileDialog.getOpenFileName(
            self,
            "Sélectionnez un modèle (un fichier .pth doit être sélectionné)"
        )
        if model_dir:
            if model_dir.endswith(".pth"):
                self._model_path_input.setText(model_dir)
                self._on_load_model()
            else:
                self._show_error("Vous devez sélectionner un fichier .pth")

    def _on_browse_destination(self) -> None:
        destination_dir = QFileDialog.getExistingDirectory(self, "Sélectionnez un dossier de sortie pour les masques et les images")
        if destination_dir:
            self.destination_path_input.setText(destination_dir)
            try:
                if self._input_PhotoModele is not None:
                    self._input_PhotoModele.setDestination(Path(self.destination_path_input.text()))
            except ValueError as e:
                self._show_error(f"Erreur de destination : {e}")

    def _on_browse_image_folder_input_(self) -> None:
        path_row_dir = QFileDialog.getExistingDirectory(self, "Sélectionnez un dossier d’entrée pour les images")
        if path_row_dir:
            self._image_folder_path_input.setText(path_row_dir)
            extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
            files = [f for f in Path(path_row_dir).iterdir() if f.suffix.lower() in extensions]
            if self.destination_path_input.text():
                self._input_PhotoModele = PhotoModele(files, Path(self.destination_path_input.text()))
            else:
                self._input_PhotoModele = PhotoModele(files)
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
        self._cleaner.setThresh(thresh)
        self._cleaner.setForce_simplificication(simplification)

        # refresh image
        if self._input_PhotoModele is None:
            return

        try:
            img, proba = self._input_PhotoModele.getActuel()
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
                default_model_source = Path(__file__).parent / "models" / "resnext50_32x4d_plusplus_4.pth"
                if not default_model_source.exists():
                    self._show_error("No .pth found in selected folder, and bundled resnext50_32x4d_plusplus_4.pth is missing.")
                    return
                self._show_error("No .pth found in selected folder, and bundled resnext50_32x4d_plusplus_4.pth is missing.")
                return
        else:
            self._show_error("Select a valid .pth model file or a folder.")
            return

        try:
            self._loading_widget.create_model(model_path, (256+128,512+256))
            self._model_path = model_path
            
        except Exception as exc:  # pragma: no cover - GUI runtime guard
            self._show_error(f"Could not load model: {exc}")
            return

        if copied_default_model:
            self._show_info(f"No .pth model was found in the folder. Copied bundled model to: {model_path}")


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
        if self._input_PhotoModele is not None:
            image_paths = self._input_PhotoModele.images

            self._loading_widget.show()
            try:
                self._loading_widget.start(image_paths)
                self._loading_widget.finished.connect(self._on_loading_finished)
                self._loading_widget.error.connect(self._show_error)
            except AttributeError as e:
                self._show_error(f"Erreur de prediction : {e}")
                self._loading_widget.hide()
            except Exception as e:
                self._show_error(f"Erreur de prediction : {e}")
                self._on_loading_finished()            
        else:
            self._show_error("Pas de Photo disponible")
        
    def _on_loading_finished(self):
        self._loading_widget.hide()
        if self._input_PhotoModele is not None:
            img, proba = self._input_PhotoModele.getActuel()
            #self.cleaner.setImageMasque(img, proba) #Pour soulager _show_image()
            self._show_image(img, proba)    
        else:
            self._show_error("Il n'y a pas d'Image disponible ... 😢")


