import os

import numpy as np
from comet_ml import Experiment
import torch.optim as optim

from AutoEncoder import ModelSaver, MyNet, test, train, Visualize
from UtilsEncoder import GetData, GetDataloader, CodeLogger

BATCH_SIZE = 128
EPOCHS = 101
# CODE_FILES = ["MainEncoder.py", "AutoEncoder.py", "UtilsEncoder.py"]
# EXPERIMENT = Experiment(
#     project_name="tumor",
#     workspace="mayank31398",
#     auto_output_logging=None,
#     auto_metric_logging=False,
#     auto_param_logging=False
# )


# def main():
#     EXPERIMENT.set_code(CodeLogger(CODE_FILES))
#     EXPERIMENT.add_tag("AutoEncoder")

#     dataset_x, dataset_y = GetData(is_train=True)
#     train_data = GetDataloader(dataset_x, dataset_y, BATCH_SIZE)

#     dataset_x, dataset_y = GetData(is_train=False)
#     test_data = GetDataloader(dataset_x, dataset_y, BATCH_SIZE)

#     model = MyNet()
#     optimizer = optim.Adam(model.parameters(), lr=0.01)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, 0.5)
#     model_saver = ModelSaver(os.path.join("Models", "AutoEncoder"))

#     for epoch in range(EPOCHS):
#         loss = train(train_data, model, optimizer, EXPERIMENT)
#         print("Training loss @ epoch", epoch, "=", loss)

#         loss = test(test_data, model, EXPERIMENT)
#         print("Validation loss @ epoch", epoch, "=", loss)
#         model_saver.Save("epoch" + str(epoch) + ".pt",
#                          model, optimizer, scheduler)
#         print()

#         if(epoch % 10 == 0):
#             scheduler.step()

def main():
    dataset_x, dataset_y = GetData(is_train=False)
    train_data = GetDataloader(dataset_x, dataset_y, BATCH_SIZE)

    model = MyNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 0.5)
    model_saver = ModelSaver(os.path.join("Models", "AutoEncoder"))
    model_saver.Load("epoch13.pt", model, optimizer, scheduler)

    Visualize(train_data, model)


if(__name__ == "__main__"):
    main()
