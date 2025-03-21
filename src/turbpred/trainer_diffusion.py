import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from turbpred.model import PredictionModel
from turbpred.loss import PredictionLoss
from turbpred.loss_history import LossHistory
from turbpred.params import DataParams, TrainingParams


class TrainerDiffusion(object):
    model: PredictionModel
    trainLoader: DataLoader
    optimizer: Optimizer
    trainHistory: LossHistory
    writer: SummaryWriter
    p_t: TrainingParams

    def __init__(self, model:PredictionModel, trainLoader:DataLoader, optimizer:Optimizer,
            trainHistory:LossHistory, writer:SummaryWriter, p_t:TrainingParams):
        self.model = model
        self.trainLoader = trainLoader
        self.optimizer = optimizer
        self.trainHistory = trainHistory
        self.writer = writer
        self.p_t = p_t


    # run one epoch of training
    def trainingStep(self, epoch:int):
        assert (len(self.trainLoader) > 0), "Not enough samples for one batch!"
        timerStart = time.perf_counter()
        timerEnd = 0

        self.model.train()
        for s, sample in enumerate(self.trainLoader, 0):
            self.optimizer.zero_grad()

            device = "cuda" if self.model.useGPU else "cpu"
            data = sample["data"].to(device)
            simParameters = sample["simParameters"].to(device) if type(sample["simParameters"]) is not dict else None

            data_0 = data[:, :2]

            prediction_0, _, _ = self.model(data_0, simParameters)
            noise_0, predictedNoise_0, predictedX0 = prediction_0[0], prediction_0[1], prediction_0[2]
            print(noise_0.shape, predictedNoise_0.shape, predictedX0.shape, data.shape)

            data_1 = data[:, 1:] # torch.cat((predictedX0, data[:,2].unsqueeze(1)), axis=1)

            prediction_1, _, _ = self.model(data_1, simParameters)
            noise_1, predictedNoise_1, _ = prediction_1[0], prediction_1[1], prediction_1[2]
            
            loss = F.smooth_l1_loss(noise_0, predictedNoise_0) + F.smooth_l1_loss(noise_1, predictedNoise_1)
            loss.backward()

            self.optimizer.step()

            timerEnd = time.perf_counter()

            lossParts = {
                "lossFull" : loss,
                "lossRecMSE" : loss,
                "lossRecLSIM" : torch.tensor([0]),
                "lossPredMSE" : torch.tensor([0]),
                "lossPredLSIM" : torch.tensor([0]),
            }
            lossSeq = {"MSE" : torch.tensor([0,0,0,0]), "LSIM" : torch.tensor([0,0,0,0])}

            self.trainHistory.updateBatch(lossParts, lossSeq, s, (timerEnd-timerStart)/60.0)

        timerEnd = time.perf_counter()
        self.trainHistory.updateEpoch((timerEnd-timerStart)/60.0)

        self.trainHistory.prepareAndClearForNextEpoch()




class TesterDiffusion(object):
    model: PredictionModel
    testLoader: DataLoader
    criterion: PredictionLoss
    testHistory: LossHistory
    p_t: TrainingParams

    def __init__(self, model:PredictionModel, testLoader:DataLoader, criterion:PredictionLoss,
                    testHistory:LossHistory, p_t:TrainingParams):
        self.model = model
        self.testLoader = testLoader
        self.criterion = criterion
        self.testHistory = testHistory
        self.p_t = p_t


    # run one epoch of testing
    def testStep(self, epoch:int):
        if epoch % self.testHistory.epochStep != self.testHistory.epochStep - 1:
            return

        assert (len(self.testLoader) > 0), "Not enough samples for one batch!"
        timerStart = time.perf_counter()
        timerEnd = 0

        self.model.eval()
        with torch.no_grad():
            for s, sample in enumerate(self.testLoader, 0):
                device = "cuda" if self.model.useGPU else "cpu"
                data = sample["data"].to(device)
                simParameters = sample["simParameters"].to(device) if type(sample["simParameters"]) is not dict else None
                if "obsMask" in sample:
                    obsMask = sample["obsMask"].to(device)
                    obsMask = torch.unsqueeze(torch.unsqueeze(obsMask, 1), 2)
                else:
                    obsMask = None

                prediction, _, _ = self.model(data, simParameters)

                if obsMask is not None:
                    _, lossParts, lossSeq = self.criterion(prediction * obsMask, data * obsMask, None, None, weighted=False, noLSIM=False)
                else:
                    _, lossParts, lossSeq = self.criterion(prediction, data, None, None, weighted=False, noLSIM=False)

                timerEnd = time.perf_counter()
                self.testHistory.updateBatch(lossParts, lossSeq, s, (timerEnd-timerStart)/60.0)

            timerEnd = time.perf_counter()
            self.testHistory.updateEpoch((timerEnd-timerStart)/60.0)

        if obsMask is not None:
            maskedPred = prediction * obsMask
            maskedData = data * obsMask
        else:
            maskedPred = prediction
            maskedData = data

        self.testHistory.writePredictionExample(maskedPred, maskedData)
        self.testHistory.writeSequenceLoss(lossSeq)

        self.testHistory.prepareAndClearForNextEpoch()
