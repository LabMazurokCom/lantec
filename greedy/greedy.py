import numpy as np
import heapq
from sortedcontainers import SortedSet
from copy import deepcopy
class GreedyProbabilisticAlgorithm:
    """data must contain vip's alerts"""
    def __init__(self, alerts, vip, deltaTimeMinus = 120, deltaTimePlus = 120):
        vipAlertsBool = alerts['number'] == vip
        vipAlerts = alerts[vipAlertsBool]
        self.vipCameras = np.array(vipAlerts['camera'])
        self.vipBigrams = np.array((np.arange(len(self.vipCameras) - 1), np.arange(1, len(self.vipCameras)))).T
        self.vipTimes = np.array(vipAlerts['time'])
        alerts = alerts[~vipAlertsBool]
        self.deltaTimeMinus = deltaTimeMinus
        self.deltaTimePlus = deltaTimePlus
        self.numberGroups = alerts.groupby('number')
        self.numberGroups.get_group(alerts.iloc[0, 0])# initializing groups
        self.suspects = self._getSuspects(alerts)

    def _getSuspects(self, dataToInspect):
        vipBigrams = self.vipBigrams
        vipTimes = self.vipTimes
        uniqueCameras = set(dataToInspect['camera'])
        groupsCamera = dataToInspect.groupby('camera')
        groupsCameraDict = {}
        prevRightCameraId = -1
        prevSuspectRight = []
        suspectRatio = {}
        suspectChain = {}
        weightSum = 0
        weights = []
        for i in range(len(vipBigrams)):
            curBigram = vipBigrams[i]
            leftCameraId = curBigram[0]
            rightCameraId = curBigram[1]
            leftCamera = self.vipCameras[leftCameraId]
            rightCamera = self.vipCameras[rightCameraId]
            if leftCamera in uniqueCameras and rightCamera in uniqueCameras:
                leftTime = vipTimes[leftCameraId]
                rightTime = vipTimes[rightCameraId]
                if leftCamera not in groupsCameraDict:
                    groupsCameraDict[leftCamera] = groupsCamera.get_group(leftCamera)
                if rightCamera not in groupsCameraDict:
                    groupsCameraDict[rightCamera] = groupsCamera.get_group(rightCamera)
                leftCameraGroup = groupsCameraDict[leftCamera]
                rightCameraGroup = groupsCameraDict[rightCamera]
                if prevRightCameraId == leftCameraId:
                    suspectLeft = prevSuspectRight
                else:
                    suspectLeft = leftCameraGroup[(leftTime - self.deltaTimeMinus <= leftCameraGroup['time']) & (
                                leftCameraGroup['time'] <= leftTime + self.deltaTimePlus)]
                suspectRight = rightCameraGroup[(rightTime - self.deltaTimeMinus <= rightCameraGroup['time']) & (
                                rightCameraGroup['time'] <= rightTime + self.deltaTimePlus)]
                intersect = np.intersect1d(suspectLeft['number'], suspectRight['number'])
                weightSum += 1/ (len(intersect) + 1)
                weights.append(1/ (len(intersect) + 1))
                for number in intersect:
                    if number not in suspectRatio:
                        suspectRatio[number] = 0
                        suspectChain[number] = []
                    suspectRatio[number] += 1 / (len(intersect) + 1)
                    suspectChain[number].append(i)
                prevSuspectRight = suspectRight
                prevRightCameraId = rightCameraId
            else:
                prevSuspectRight = []
        setSuspectChain = {}
        for k, v in suspectChain.items():
            setSuspectChain[k] = SortedSet(v)
        return setSuspectChain, suspectRatio, weights


    def _createSuspectSequence(self, suspectChain):
        lastCam = -1
        a = []
        for b in suspectChain:
            leftCamId = self.vipBigrams[b][0]
            rightCamId = self.vipBigrams[b][1]
            if self.vipCameras[leftCamId] != self.vipCameras[rightCamId] or self.vipTimes[rightCamId] - self.vipTimes[leftCamId] >= self.deltaTimeMinus + self.deltaTimePlus:
                if lastCam != leftCamId:
                    a.append(leftCamId)
                a.append(rightCamId)
                lastCam = rightCamId
        return a

    def findVehicleMax(self, suspectChain, suspectRatio):
        sequences = {}
        for vehicle, chain in suspectChain.items():
            suspectSequence = self._createSuspectSequence(chain)
            if len(suspectSequence) == 0:
                suspectRatio[vehicle] = 0
            sequences[vehicle] = suspectSequence
        heap = [(-value, key) for key, value in suspectRatio.items()]
        heapq.heapify(heap)
        maxNumber = None
        maxRatio = -np.inf
        i = 0
        run = True
        while run and len(heap) > 0:
            i += 1
            score, number = heapq.heappop(heap)
            score = -score
            if score >= maxRatio and len(sequences[number]) > 0:
                supposedMaxRatio = score / self.getVehicleTypicalRate(sequences[number], self.numberGroups.get_group(number)['camera'])
                if supposedMaxRatio > maxRatio:
                    maxRatio = supposedMaxRatio
                    maxNumber = number
            else:
                run = False
        if maxNumber:
            return maxNumber, suspectChain[maxNumber], sequences[maxNumber]
        else:
            return None, None, None

    def getVehicleTypicalRate(self, sequence, fullRoute):
        return 1
        vipCameras = self.vipCameras
        sequence = np.array(sequence, dtype=int)
        fullRoute = np.array(fullRoute)
        partInd = 0
        count = 0
        for i in fullRoute:
            if vipCameras[sequence[partInd]] == i:
                partInd += 1
            if partInd == len(sequence) - 1:
                count += 1
                partInd = 0
        return count

    def findCover(self, maxVehicleNumber, suspects = None, threshold=0.95):
        cover = []
        if suspects is None:
            suspects = deepcopy(self.suspects)
        else:
            suspects = deepcopy(suspects)
        vipBigrams = self.vipBigrams
        toCover = len(vipBigrams)
        covered = 0
        suspectChain, suspectRatio, weights = suspects
        for i in range(maxVehicleNumber):
            vehicleMax, bigrams, sequence = self.findVehicleMax(suspectChain, suspectRatio)
            if vehicleMax is not None:
                cover.append({'number': vehicleMax, 'alerts': [{'camera': self.vipCameras[i], 'time': self.vipTimes[i]} for i in sequence]})
                for k, v in suspectChain.items():
                    if k != vehicleMax:
                        for b in bigrams:
                            if b in v:
                                v.remove(b)
                                suspectRatio[k] -= weights[b]
                del suspectChain[vehicleMax]
                del suspectRatio[vehicleMax]
                covered += len(bigrams)
            if vehicleMax is None or toCover * threshold <= covered:
                break
        return {'cover': cover, 'coverRatio': covered / toCover}

    def findCovers(self, maxCoverNumber, maxVechicleNumber):
        covers = []
        suspects = deepcopy(self.suspects)
        for i in range(maxCoverNumber):
            cover = self.findCover(maxVechicleNumber, suspects)
            if len(cover['cover']) > 0:
                covers.append(cover)
                print(cover)
                del suspects[0][cover['cover'][0]['number']]
                del suspects[1][cover['cover'][0]['number']]
            else:
                break
        return covers

