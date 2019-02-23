import os

import numpy as np
from comet_ml import Experiment
import torch.optim as optim

# from FC import ModelSaver, MyNet, test, train
from FC import ModelSaver, MyNet, test, train
from utils import GetData, GetDataloader, CodeLogger

BATCH_SIZE = 128
EPOCHS = 101
CODE_FILES = ["main.py", "utils.py", "FC.py"]
EXPERIMENT = Experiment(
    project_name="tumor",
    workspace="mayank31398",
    auto_output_logging=None,
    auto_metric_logging=False,
    auto_param_logging=False
)


def main():
    EXPERIMENT.set_code(CodeLogger(CODE_FILES))
    EXPERIMENT.add_tag("FC")

    dataset_x, dataset_y = GetData(is_train=True)
    train_data = GetDataloader(dataset_x, dataset_y, BATCH_SIZE)

    dataset_x, dataset_y = GetData(is_train=False)
    test_data = GetDataloader(dataset_x, dataset_y, BATCH_SIZE)

    model = MyNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 0.5)
    model_saver = ModelSaver(os.path.join("Models", "FC"))

    for epoch in range(EPOCHS):
        loss, accuracy, f1score, precision, recall = train(
            train_data, model, optimizer, EXPERIMENT)
        print("Training loss @ epoch", epoch, "=", loss)
        print("Training accuracy @ epoch", epoch, "=", accuracy)
        print("Training F1 score @ epoch", epoch, "=", f1score)
        print("Training precision @ epoch", epoch, "=", precision)
        print("Training recall @ epoch", epoch, "=", recall)
        with EXPERIMENT.train():
            EXPERIMENT.log_metric("Batch loss", loss, step=epoch)
            EXPERIMENT.log_metric("Batch accuracy", accuracy, step=epoch)
            EXPERIMENT.log_metric("Batch F1 score", f1score, step=epoch)
            EXPERIMENT.log_metric("Batch precision", precision, step=epoch)
            EXPERIMENT.log_metric("Batch recall", recall, step=epoch)

        loss, accuracy, f1score, precision, recall = test(
            test_data, model, EXPERIMENT)
        print("Validation loss @ epoch", epoch, "=", loss)
        print("Validation accuracy @ epoch", epoch, "=", accuracy)
        print("Validation F1 score @ epoch", epoch, "=", f1score)
        print("Validation precision @ epoch", epoch, "=", precision)
        print("Validation recall @ epoch", epoch, "=", recall)
        with EXPERIMENT.test():
            EXPERIMENT.log_metric("Batch loss", loss, step=epoch)
            EXPERIMENT.log_metric("Batch accuracy", accuracy, step=epoch)
            EXPERIMENT.log_metric("Batch F1 score", f1score, step=epoch)
            EXPERIMENT.log_metric("Batch precision", precision, step=epoch)
            EXPERIMENT.log_metric("Batch recall", recall, step=epoch)
        print("###############################################\n")
        model_saver.Save("epoch" + str(epoch) + ".pt",
                         model, optimizer, scheduler)

        if(epoch % 10 == 0):
            scheduler.step()


if(__name__ == "__main__"):
    main()
