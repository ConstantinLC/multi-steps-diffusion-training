import os
import copy
from typing import Dict
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, SubsetRandomSampler

from turbpred.model import PredictionModel
from turbpred.logger import Logger
from turbpred.params import DataParams, TrainingParams, LossParams, ModelParamsEncoder, ModelParamsDecoder, ModelParamsLatent
from turbpred.turbulence_dataset import KolmogorovDataset_Rozet
from turbpred.data_transformations import Transforms
from turbpred.loss import PredictionLoss
from turbpred.loss_history import LossHistory
from turbpred.trainer_diffusion import TrainerDiffusion, TesterDiffusion

from math import sqrt

if __name__ == "__main__":
    useGPU = True
    gpuID = "3"

    #torch.manual_seed(1)
    #torch.cuda.manual_seed(1)

    resolution = "64"
    timeStep = 1
    framesPerTimestep = 1
    modelDim = 64
    diffSchedule = "quadratic"
    sequenceLength=[2,timeStep]

    ### ACDM
    arch= "direct-ddpm"
    modelName = f"2D_Kolmo_rozet_multi-steps/res_{resolution}_schedule_{diffSchedule}_sequenceLength_{sequenceLength[0]}_ts_{timeStep}_dim_{modelDim}_framesPerTs_{framesPerTimestep}_arch_{arch}"
    p_d = DataParams(batch=32, augmentations=[], sequenceLength=sequenceLength, randSeqOffset=True,
                dataSize=[modelDim,modelDim], dimension=2, simFields=[], simParams=[], normalizeMode=None)
    p_t = TrainingParams(epochs=5000, lr=0.0002)
    p_l = LossParams()
    p_me = None
    p_md = ModelParamsDecoder(arch=arch, diffSteps=100, diffSchedule=diffSchedule, diffCondIntegration="clean", trainingNoise=0)
    p_ml = None
    pretrainPath = ""

    ### ACDM_ncn
    # modelName = "2D_Iso/128_acdm-r100_ncn"
    # p_d = DataParams(batch=64, augmentations=["normalize"], sequenceLength=[3,1], randSeqOffset=True,
    #             dataSize=[128,64], dimension=2, simFields=["velZ", "pres"], simParams=[], normalizeMode="isoSingle")
    # p_t = TrainingParams(epochs=100, lr=0.0001)
    # p_l = LossParams()
    # p_me = None
    # p_md = ModelParamsDecoder(arch="direct-ddpm+Prev", diffSteps=100, diffSchedule="linear", diffCondIntegration="clean", trainingNoise=0.0)
    # p_ml = None
    # pretrainPath = ""

    
    ### Refiner-r4_std0.00001
    """modelName = f"2D_Kolmo_rozet_upsampled/res_{resolution}_ts_{timeStep}_dim_{modelDim}_framesPerTs_{framesPerTimestep}_128_refiner-r4_std0.00001"
    p_d = DataParams(batch=64, augmentations=[], sequenceLength=[2,timeStep], randSeqOffset=True,
                 dataSize=[modelDim,modelDim], dimension=2, simFields=[], simParams=[], normalizeMode=None)
    p_t = TrainingParams(epochs=5000, lr=0.0001)
    p_l = LossParams()
    p_me = None
    p_md = ModelParamsDecoder(arch="refiner", diffSteps=3, refinerStd=sqrt(1e-3))
    p_ml = None
    pretrainPath = """""


    trainSet = KolmogorovDataset_Rozet("Training", "../sda/data", mode="train", resolution=resolution, sequenceLength=p_d.sequenceLength, framesPerTimeStep=framesPerTimestep)

    testSets = {"test": KolmogorovDataset_Rozet("Test", "../sda/data", mode="valid", resolution=resolution, sequenceLength=[64, timeStep], framesPerTimeStep=1)}
   
    # Test set length 172


def train(modelName:str, trainSet:KolmogorovDataset_Rozet, testSets:Dict[str,KolmogorovDataset_Rozet],
        p_d:DataParams, p_t:TrainingParams, p_l:LossParams, p_me:ModelParamsEncoder, p_md:ModelParamsDecoder,
        p_ml:ModelParamsLatent, pretrainPath:str="", useGPU:bool=True, gpuID:str="0"):

    # DATA AND MODEL SETUP
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuID
    logger = Logger(modelName, addNumber=True)
    model = PredictionModel(p_d, p_t, p_l, p_me, p_md, p_ml, pretrainPath, useGPU)
    #model = PredictionModel.load("/mnt/SSD2/constantin/autoreg-pde-diffusion/runs/2D_Kolmo_rozet/res_64_schedule_linear_ts_1_dim_64_framesPerTs_1_acdm-r100_20/Model.pth")
    model.printModelInfo()
    criterion = PredictionLoss(p_l, p_d.dimension, p_d.simFields, useGPU)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p_t.lr, weight_decay=p_t.weightDecay)
    logger.setup(model, optimizer)

    #transTrain = Transforms(p_d)
    #trainSet.transform = transTrain
    #trainSet.printDatasetInfo()
    trainSampler = RandomSampler(trainSet)
    #trainSampler = SubsetRandomSampler(range(2))
    trainLoader = DataLoader(trainSet, sampler=trainSampler,
                    batch_size=p_d.batch, drop_last=True, num_workers=4)
    trainHistory = LossHistory("_train", "Training", logger.tfWriter, len(trainLoader),
                                    0, 1, printInterval=1, logInterval=1, simFields=p_d.simFields)
    trainer = TrainerDiffusion(model, trainLoader, optimizer, trainHistory, logger.tfWriter, p_t)

    #print(torch.load("/mnt/SSD2/constantin/autoreg-pde-diffusion/runs/2D_Kolmo_rozet/res_64_schedule_linear_ts_1_dim_64_framesPerTs_1_acdm-r100_20/TrainState.pth"))
    #loaded = torch.load("/mnt/SSD2/constantin/autoreg-pde-diffusion/runs/2D_Kolmo_rozet_upsampled/res_64_schedule_psd_periodicconv_ts_1_dim_64_framesPerTs_1_arch_direct-ddpm+Prev_14/Model.pth", map_location=torch.device('cpu'))

    #model.modelDecoder.load_state_dict(loaded["stateDictDecoder"])

    testers = []
    testHistories = []
    for shortName, testSet in testSets.items():
        p_d_test = copy.deepcopy(p_d)
        #p_d_test.augmentations = ["normalize"]
        p_d_test.sequenceLength = testSet.sequenceLength
        p_d_test.randSeqOffset = False
        p_d_test.batch = 102
        #if p_d.sequenceLength[0] != p_d_test.sequenceLength[0]:
        #    p_d_test.batch = 1

        #transTest = Transforms(p_d_test)
        #testSet.transform = transTest
        #testSet.printDatasetInfo()
        testSampler = SequentialSampler(testSet)
        #testSampler = SubsetRandomSampler(range(2))
        testLoader = DataLoader(testSet, sampler=testSampler,
                        batch_size=p_d_test.batch, drop_last=False, num_workers=4)
        testHistory = LossHistory(shortName, testSet.name, logger.tfWriter, len(testLoader),
                                    20, 20, printInterval=0, logInterval=0, simFields=p_d.simFields)
        tester = TesterDiffusion(model, testLoader, criterion, testHistory, p_t)
        testers += [tester]
        testHistories += [testHistory]

    #if loadEpoch > 0:
    #    logger.resumeTrainState(loadEpoch)

    # TRAINING
    print('Starting Training')
    logger.saveTrainState(0)

    for tester in testers:
        tester.testStep(0)

    for epoch in range(0, p_t.epochs):
        trainer.trainingStep(epoch)
        logger.saveTrainState(epoch, checkpointEvery=20)

        for tester in testers:
            tester.testStep(epoch+1)

        trainHistory.updateAccuracy([p_d,p_t,p_l,p_me,p_md,p_ml], testHistories, epoch==p_t.epochs-1)

    logger.close()

    print('Finished Training')


if __name__ == "__main__":
    train(modelName, trainSet, testSets, p_d, p_t, p_l, p_me, p_md, p_ml, pretrainPath=pretrainPath, useGPU=useGPU, gpuID=gpuID) #type:ignore