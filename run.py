# from model.layers_LaBSE_neighbor import Trainer
# from model.layers_LaBSE_SSL import Trainer
from model.train import Trainer



if __name__ == '__main__':
    trainer = Trainer(seed=37)
    trainer.train(0)
