from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist


class VehicleTracker():

    def __init__(self, detection_zone=(0, 0, 200, 200), maxDisappeared=3, min_path=4):

        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.path_length = {}
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
        self.count = 0
        self.min_path = min_path
        self.detection_zone = detection_zone

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.path_length[self.nextObjectID] = 0
        self.nextObjectID += 1

    def check_detection_zone(self, objectID):
        centroid = self.objects[objectID][1]
        if (self.detection_zone[0] < centroid[0] < self.detection_zone[2] and self.detection_zone[1] < centroid[1] <
                self.detection_zone[3]):
            return True
        return False

    def deregister(self, objectID):
        if (self.path_length[objectID] >= self.min_path and self.check_detection_zone(objectID)):
            self.count += 1
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if (len(rects) == 0):
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if (self.disappeared[objectID] > self.maxDisappeared):
                    self.deregister(objectID)
            return self.objects

        if (len(self.objects) == 0):
            for i in range(0, len(rects)):
                self.register(rects[i])
        else:
            objectIDs = list(self.objects.keys())
            objectPosition = list(self.objects.values())

            D = dist.cdist(np.array([box[1] for box in objectPosition]), np.array([box[1] for box in rects]))

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = rects[col]
                self.path_length[objectID] += 1
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(rects[col])
        return self.objects
