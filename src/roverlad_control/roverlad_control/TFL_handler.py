import onnxruntime as ort
import numpy as np
import cv2

class TFLightInferenceRunner:
    def __init__(self, modelPath):
        print(ort.__version__)
        print(ort.get_available_providers())

        self.session = ort.InferenceSession(modelPath, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.anchors = np.array([[10, 13], [16, 30], [33, 23]], dtype=np.float32)
        
        self.inputName = self.session.get_inputs()[0].name
        self.outputName = self.session.get_outputs()[0].name        

        print("Inputs:", self.session.get_inputs()[0].shape)
        print("Outputs:", self.session.get_inputs()[0].type)
        print("Model providers:", self.session.get_providers())

        print("> CVInferenceRunner Up <")


    def run(self, image, conf=0.85, debug=False):
        imageT = self.transformImg(image)

        if (debug):
            print("Before Run")
        
        output = self.session.run(None, {self.inputName: imageT})
        
        if (debug):
            print("After Run")        
            print("Output shape:", [o.shape for o in output])

        pred = output[0]
        
        # sigmoid
        pred[..., 0:2] = 1 / (1 + np.exp(-pred[..., 0:2]))
        pred[..., 4:]  = 1 / (1 + np.exp(-pred[..., 4:]))
        
        # shape
        batch, gridH, gridW, numAnchors, predSize = pred.shape

        if (debug):
            print("Got Shapes")        

        gridX, gridY = self.getGridXY(gridW, gridH)        
        bx, by, bw, bh = self.getBorders(numAnchors, pred, gridX, gridY, gridW, gridH)        
        classPred, finalConf = self.getPredConf(pred)
        
        if (debug):
            print("Got GXY, borders, pred conf")        

        mask = finalConf > conf
        
        bx = bx[mask]
        by = by[mask]
        bw = bw[mask]
        bh = bh[mask]
        scores = finalConf[mask]
        labels = classPred[mask]
        
        if (debug):
            print("masks passed")        

        boxes_nms, scores_nms, labels_nms = CVHelpers.getBatchedNMS(scores, labels, bx, by, bw, bh)
        boxesNP, scoresNP, labelsNP = self.mergeBoxesByLabel(boxes_nms, scores_nms, labels_nms)
                  
        if (debug):
            print("Boxes:", boxes_nms)
            print("Scores:", scores_nms)
            print("Labels:", labels_nms)
        
        boxesPX = self.getBoxesPx(boxesNP, image)        

        return boxesNP, boxesPX, scoresNP, labelsNP

    def transformImg(self, img):
        image = cv2.resize(img, (512, 512))
        image = image.transpose(2, 0, 1)
        image = image / 255.0
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)

        return image

    def getPredConf(self, pred):
        conf = pred[..., 4]
        classProbs = pred[..., 5:]
        
        classPred = np.argmax(classProbs, axis=-1)
        classConf = np.max(classProbs, axis=-1)
        
        finalConf = conf * classConf
    
        return classPred, finalConf
    
    def getGridXY(self, gridW, gridH):
        gridX = np.tile(np.arange(gridW), (gridH, 1))
        gridY = gridX.T
        
        gridX = gridX.reshape(1, gridH, gridW, 1)
        gridY = gridY.reshape(1, gridH, gridW, 1)
    
        return gridX, gridY
    
    def getBorders(self, numAnchors, pred, gridX, gridY, gridW, gridH):
        anchors = self.anchors.reshape(1, 1, 1, numAnchors, 2)
    
        bx = (pred[..., 0] + gridX) / gridW
        by = (pred[..., 1] + gridY) / gridH
        
        bw = anchors[..., 0] * np.exp(pred[..., 2]) / 512
        bh = anchors[..., 1] * np.exp(pred[..., 3]) / 512
    
        return bx, by, bw, bh
    
    def getBoxesPx(self, boxes, image):
        if boxes.size == 0:
            return boxes  # return empty safely
    
        h, w, _ = image.shape
    
        boxesPx = boxes.copy()
        boxesPx[:, 0] *= w
        boxesPx[:, 1] *= h
        boxesPx[:, 2] *= w
        boxesPx[:, 3] *= h
    
        return boxesPx.astype(int)
    
    def mergeBoxesByLabel(self, boxes, scores, labels):
        mergedBoxes = []
        mergedScores = []
        mergedLabels = []
    
        for cls in np.unique(labels):
            mask = labels == cls
    
            clsBoxes = boxes[mask]
            clsScores = scores[mask]
    
            if len(clsBoxes) == 0:
                continue
    
            x1 = clsBoxes[:, 0].min()
            y1 = clsBoxes[:, 1].min()
            x2 = clsBoxes[:, 2].max()
            y2 = clsBoxes[:, 3].max()
    
            mergedBoxes.append([x1, y1, x2, y2])
            mergedScores.append(clsScores.max())
            mergedLabels.append(cls)
    
        return (np.array(mergedBoxes), np.array(mergedScores), np.array(mergedLabels))
               

class CVHelpers:

    @staticmethod
    def getBatchedNMS(scores, labels, bx, by, bw, bh):
        x1 = bx - bw / 2
        y1 = by - bh / 2
        x2 = bx + bw / 2
        y2 = by + bh / 2
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
    
        keepAll = []
    
        for cls in np.unique(labels):
            mask = labels == cls
    
            clsBoxes = boxes[mask]
            clsScores = scores[mask]
            clsIdx = np.where(mask)[0]
    
            keep = CVHelpers.nms(clsBoxes, clsScores, 0.5)
            keepAll.extend(clsIdx[keep])
    
        keepAll = np.array(keepAll, dtype=np.int64)
    
        return boxes[keepAll], scores[keepAll], labels[keepAll]
    
    @staticmethod
    def nms(boxes, scores, iouThresh=0.5):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
    
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
    
        keep = []
    
        while order.size > 0:
            i = order[0]
            keep.append(i)
    
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
    
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
    
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
    
            inds = np.where(iou <= iouThresh)[0]
            order = order[inds + 1]
    
        return np.array(keep)
    