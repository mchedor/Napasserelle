from qtpy.QtCore import Signal, Qt, QTimer
from qtpy.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton


class EditWidget(QWidget):
    modeChanged = Signal(bool)
    valuesChanged = Signal(int, float)

    def __init__(self, cleaner, parent=None):
        super().__init__(parent)

        self.cleaner = cleaner
        self._edit_pred_mode = False


        self._emit_timer = QTimer(self)
        self._emit_timer.setSingleShot(True)
        self._emit_timer.setInterval(750)

        self._emit_timer.timeout.connect(self._emit_values_now)

        # --- LABELS ---
        self.threshold_label = QLabel("Threshold")
        self.simplification_label = QLabel("Simplification")

        # --- SLIDERS ---
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(self.cleaner.thresh)

        self.simplification_slider = QSlider(Qt.Orientation.Horizontal)
        self.simplification_slider.setRange(0, 23)
        self.simplification_slider.setValue(12)

        # --- BUTTON ---
        self.mode_button = QPushButton("Modifier la prediction")
        self.mode_button.clicked.connect(self._toggle_mode)

        # --- LAYOUT ---
        layout = QVBoxLayout(self)
        layout.addWidget(self.mode_button)
        layout.addWidget(self.threshold_label)
        layout.addWidget(self.threshold_slider)
        layout.addWidget(self.simplification_label)
        layout.addWidget(self.simplification_slider)

        self._update_ui_state()

        # signals
        self.threshold_slider.valueChanged.connect(self._emit_values)
        self.simplification_slider.valueChanged.connect(self._emit_values)

        # init labels
        self._update_labels()

    # -------------------------
    # MODE
    # -------------------------

    def _toggle_mode(self):
        self._edit_pred_mode = not self._edit_pred_mode

        if self._edit_pred_mode:
            self.mode_button.setText("Appliquer les modifications")
        else:
            self.mode_button.setText("Modifier la prediction")

        self._update_ui_state()
        self.modeChanged.emit(self._edit_pred_mode)

    def _update_ui_state(self):
        self.threshold_slider.setEnabled(self._edit_pred_mode)
        self.simplification_slider.setEnabled(self._edit_pred_mode)

        # optionnel mais recommandé : griser aussi les labels
        self.threshold_label.setEnabled(self._edit_pred_mode)
        self.simplification_label.setEnabled(self._edit_pred_mode)

    # -------------------------
    # SIMPLIFICATION
    # -------------------------

    def compute_simplification(self, v: int) -> float:
        if v <= 0:
            return 0.0

        base = [4, 5, 6, 7, 8, 9, 1, 2, 3][v % 9]
        scale = 10 ** (((v + 3) // 9) - 4)

        return base * scale

    # -------------------------
    # VALUES
    # -------------------------

    def _emit_values(self):
        if not self._edit_pred_mode:
            return
        
        # update labels
        self._update_labels()

        # restart timer à chaque mouvement
        self._emit_timer.start()

    def _emit_values_now(self):
        if not self._edit_pred_mode:
            return

        thresh = self.threshold_slider.value()
        simpl = self.compute_simplification(self.simplification_slider.value())

        # update labels
        self._update_labels()

        # emit signal
        self.valuesChanged.emit(thresh, simpl)

    def _update_labels(self):
        self.threshold_label.setText(f"Threshold: {self.threshold_slider.value()}")
        self.simplification_label.setText(
            f"Simplification: {self.compute_simplification(self.simplification_slider.value()):.6f}"
        )