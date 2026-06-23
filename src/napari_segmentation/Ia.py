from pathlib import Path
import traceback




# ---------------------------
# Thread pour simuler une tâche longue
# ---------------------------
class UnetWorker():

    def __init__(self, model_path, resolution=(256+128, 512+256), batch_size=8):
        super().__init__()
        
        self.model = None
        self.lib = False
        self.resolution = resolution
        self.batch_size = batch_size
        self.load_model(model_path)

    # -------------------------
    # 1. LOAD MODEL
    # -------------------------
    def load_model(self, model_path):

        if self.lib == False:
            import torch           
            import cv2
            import numpy as np
            from segmentation_models_pytorch import UnetPlusPlus
            import albumentations as A

            # expose libs en attributs si besoin ailleurs
            self.torch = torch
            self.cv2 = cv2
            self.np = np
            self.A = A
            self.lib = True

            self.device = "cpu"#torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self.transform = A.Compose([
                A.Resize(height=self.resolution[0], width=self.resolution[1]),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                )
            ]) 
        
        self.model = UnetPlusPlus( # type: ignore
            encoder_name="resnext50_32x4d",
            encoder_weights=None,
            in_channels=3,
            classes=1
        ).to(self.device)

        self.model.load_state_dict(self.torch.load(model_path, map_location=self.device,weights_only=True))
        self.model.eval()

    # -------------------------
    # 2. PREPROCESS SINGLE IMAGE
    # -------------------------
    def _preprocess(self, image):
        augmented = self.transform(image=image)
        image = augmented["image"]  # HWC float32
        tensor = self.torch.from_numpy(image).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

    # -------------------------
    # 4. INFERENCE
    # -------------------------
    def _predict(self, input):
        """_summary_

        Parameters
        ----------
        input : 
            batch_tensor ou image

        Returns
        -------
        _type_
            prediction
        """
        if self.model is None:
            raise RuntimeError("Aucun model n'est defini")

        with self.torch.no_grad():
            preds = self.model(input/255)
            preds = self.torch.sigmoid(preds)
        mask = preds.squeeze().cpu().numpy()
        mask = (mask * 255).astype(self.np.uint8)
        return mask

    # -------------------------
    # 5. POSTPROCESS SINGLE MASK
    # -------------------------
    def _postprocess(self, pred):
        """Retourne la carte de probabilité de façon exploitable.

        Parameters
        ----------
        pred : np.ndarray
            prediction

        Returns
        -------
        np.ndarray
            [0->255]
        """
        pred = self.np.squeeze(pred)
        #pred = (pred * 255).astype(self.np.uint8)

        return pred
    
    def run_one_inference(self, img):
        img = self._preprocess(img)
        pred = self._predict(img)
        pred = self._postprocess(pred)
        return pred


    def _make_batch(self, images):

        tensor = self.torch.from_numpy(self.np.stack(images))
        tensor = tensor.permute(0, 3, 1, 2).float().to(self.device)

        return tensor

    def run_inferences(self, img_iterable, function_traitement):
        i=0
        images = []
        for img in img_iterable:
            if i < self.batch_size:
                i+=1
                img = self._preprocess(img)
                images.append(img)
            else:
                batch_tensor = self._make_batch(images)
                images = []
                i=0
                preds = self._predict(batch_tensor)
                for pred in preds:
                    self._postprocess(pred)
                    function_traitement(preds)





