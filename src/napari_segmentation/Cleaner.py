
import cv2
import numpy as np


#TODO : pour plus de rapidité : regarder si il vaut mieu pas faire le resize une fois pour toute (et donc avant les modifs) ou vos mieu le faire apres le clean
#TODO : pourquoi pas une resolution personalisée (on ne calcul que ce que l'on voit sauf pour enregistrer) => plus de fluidité.

class CleanerModele() :
    
    def __init__(self):

        self.masque  = None
        self.resolution : tuple[int, int] = (512+256,256+128)
        self.reset()
    
    def reset(self):
        self.interpolation : int = cv2.INTER_NEAREST

        self.blur_size : int   = 3
        self.ker_size : int  = 10
        self.clean_open : bool  = True
        self.clean_close : bool  = True
        self.dilate : bool  = False
        self.erode : bool  = False
        self.ker_dilateErode_size : int = 5
        self.dilateErode_iteration : int = 1 

        self.thresh : int = 128
        self.force_simplificication : float = 0.02
        

        self.invert_masque : bool = True
        self.lightmode : bool = True

    def setMasque(self, newMasque):
        self.masque = newMasque
        #self._update_image()
    
    def setMasqueResImg(self, newMasque, newImage):
        self.resolution = ((newImage.shape[1]),(newImage.shape[0]))
        #self.img = newImage
        self.masque = newMasque
        #self._update_image()

    def setLightmode(self, newLightmode : bool):
        self.lightmode = newLightmode
        #self._update_image()

    def setBlur_size(self, newBlur_size):
        self.blur_size = newBlur_size
        #self._update_image()

    def setKer_size(self, newKer_size):
        self.ker_size = newKer_size
        #self._update_image()    

    def setKer_dilateErode_size(self, newKer_size):
        self.ker_dilateErode_size = newKer_size
        #self._update_image()   

    def setDilateErode_iteration(self, newDilateErode_iteration):
        self.dilateErode_iteration = newDilateErode_iteration
        #self._update_image()   
    
    def setClean_open(self, newClean_open : bool):
        self.clean_open = newClean_open
        #self._update_image()
    
    def setClean_close(self, newClean_close : bool):
        self.clean_close = newClean_close
        #self._update_image()

    def setDilateErode(self, dialte : bool, erode : bool):
        self.dilate = dialte
        self.erode = erode
        assert (not (dialte and erode)) 
        #self._update_image()

    def setThresh(self, newThresh):
        self.thresh = newThresh
        #self._update_image()

    def setForce_simplificication(self, newForce_simplificication):
        self.force_simplificication = newForce_simplificication 
        #self._update_image()

    
    def setInvert_masque(self, newInvert_masque):
        self.invert_masque = newInvert_masque
        #self._update_image()
    


    def _simplify_mask(self, mask, epsilon_factor=0.01):
        """
        Simplifie les contours d'un masque binaire avec Douglas-Peucker.

        Parameters
        ----------
        mask : np.ndarray
            Masque binaire (0/255 ou booléen).
        epsilon_factor : float
            Facteur de simplification relatif au périmètre.

        Returns
        -------
        np.ndarray
            Masque binaire simplifié de la même taille. [0/1]
        """

        # Conversion en uint8
        #mask = (mask > 0).astype(np.uint8) * 255

        # Extraction des contours
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE
        )

        # Masque de sortie
        simplified_mask = np.zeros_like(mask)

        # Simplification et remplissage
        for contour in contours:
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            cv2.drawContours(
                simplified_mask,
                [approx],
                contourIdx=-1,
                color=1,
                thickness=cv2.FILLED
            )

        return simplified_mask
    
    def _simplify_mask_polygon(self, mask, epsilon_factor=0.01):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            #cv2.RETR_TREE,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        polygons = []

        for contour in contours:
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            poly = approx.reshape(-1, 2)[:, ::-1]

            if len(poly) >= 3:
                polygons.append(poly)

        return polygons
    
    def clean_masque(self, masque):
        masque_clean = masque.copy()
        kernel = np.ones((self.ker_size, self.ker_size), np.uint8)
        kernel_erodeDilate = np.ones((self.ker_dilateErode_size, self.ker_dilateErode_size), np.uint8)
        
        if self.blur_size > 0 :
            masque_clean = cv2.GaussianBlur(masque_clean, (self.blur_size, self.blur_size), 0)
        # masque -> 0/1
        masque_clean = (masque_clean > self.thresh).astype(np.uint8)
        #(retVal, newImg) = cv2.threshold(newImg, 130, 255, cv2.THRESH_BINARY)
        #cv2.threshold(gray_im,150,255,cv2.THRESH_BINARY_INV)
        if self.clean_open:
            masque_clean = cv2.morphologyEx(masque_clean, cv2.MORPH_OPEN, kernel)
        if self.clean_close:
            masque_clean = cv2.morphologyEx(masque_clean, cv2.MORPH_CLOSE, kernel)

        if self.erode:
            masque_clean = cv2.erode(masque_clean, kernel_erodeDilate, iterations=self.dilateErode_iteration)
        if self.dilate:
            masque_clean = cv2.dilate(masque_clean, kernel_erodeDilate, iterations=self.dilateErode_iteration)    
        
        masque_clean = cv2.resize(masque_clean, dsize=self.resolution, interpolation=self.interpolation)
        
        if self.force_simplificication > 0:
            masque_clean = self._simplify_mask(masque_clean, self.force_simplificication)
        
        
        
        return masque_clean
    

    def clean_masque_polygon(self, masque):
        masque_clean = masque.copy()
        kernel = np.ones((self.ker_size, self.ker_size), np.uint8)
        kernel_erodeDilate = np.ones((self.ker_dilateErode_size, self.ker_dilateErode_size), np.uint8)
        
        if self.blur_size > 0 :
            masque_clean = cv2.GaussianBlur(masque_clean, (self.blur_size, self.blur_size), 0)
        # masque -> 0/1
        masque_clean = (masque_clean > self.thresh).astype(np.uint8)
        #(retVal, newImg) = cv2.threshold(newImg, 130, 255, cv2.THRESH_BINARY)
        #cv2.threshold(gray_im,150,255,cv2.THRESH_BINARY_INV)
        if self.clean_open:
            masque_clean = cv2.morphologyEx(masque_clean, cv2.MORPH_OPEN, kernel)
        if self.clean_close:
            masque_clean = cv2.morphologyEx(masque_clean, cv2.MORPH_CLOSE, kernel)

        if self.erode:
            masque_clean = cv2.erode(masque_clean, kernel_erodeDilate, iterations=self.dilateErode_iteration)
        if self.dilate:
            masque_clean = cv2.dilate(masque_clean, kernel_erodeDilate, iterations=self.dilateErode_iteration)    
    
        masque_clean = cv2.resize(masque_clean, dsize=self.resolution, interpolation=self.interpolation)

        polygons = self._simplify_mask_polygon(masque_clean, self.force_simplificication)
        
        return polygons     


    def getMasqueClean(self):
        return self.clean_masque(self.masque)
    
    def getMasquePolygon(self):
        return self.clean_masque_polygon(self.masque)