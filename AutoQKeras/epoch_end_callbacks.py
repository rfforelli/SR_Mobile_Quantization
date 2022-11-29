from utils import ProgressBar
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback
import math
import numpy as np

class Epoch_End_Callback(Callback):
    def __init__(self, val_data, train_data, lg, val_step):
        super(Epoch_End_Callback, self).__init__()
        self.val_step = val_step
        self.val_data = val_data
        self.train_data = train_data
        self.lg = lg
        self.best_psnr = 0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs):
        
        self.train_data.shuffle()
        if epoch % self.val_step != 0:
            return

        # validate
        psnr = 0.0
        pbar = ProgressBar(len(self.val_data))
        for i, (lr, hr) in enumerate(self.val_data):
            sr = self.model(lr)
            sr_numpy = K.eval(sr)
            psnr += self.calc_psnr((sr_numpy).squeeze(), (hr).squeeze())
            pbar.update('')
        psnr = round(psnr / len(self.val_data), 4)
        loss = round(logs['loss'], 4)

        # save best status
        if psnr > self.best_psnr:
            self.best_psnr = psnr
            self.best_epoch = epoch

        self.lg.info('Epoch: {:4} | PSNR: {:.2f} | Loss: {:.4f} | lr: {:.2e} | Best_PSNR: {:.2f} in Epoch [{}]'.format(epoch, psnr, loss, K.get_value(self.model.optimizer.lr), self.best_psnr, self.best_epoch))

    def calc_psnr(self, y, y_target):
        h, w, c = y.shape
        y = np.clip(np.round(y), 0, 255).astype(np.float32)
        y_target = np.clip(np.round(y_target), 0, 255).astype(np.float32)

        # crop 1
        y_cropped = y[1:h-1, 1:w-1, :]
        y_target_cropped = y_target[1:h-1, 1:w-1, :]
        
        mse = np.mean((y_cropped - y_target_cropped) ** 2)
        if mse == 0:
            return 100
        return 20. * math.log10(255. / math.sqrt(mse))