from keras.callbacks import Callback


class CustomModelCheckpoint(Callback):
    """
    Callback ذخیره دوره‌ای مدل، دقیقا مثل نوت‌بوک
    """

    def __init__(self, save_path, save_freq=5, monitor="loss", save_best_only=True):
        super().__init__()
        self.save_path = save_path
        self.save_freq = save_freq
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best = None
        self.epochs_since_save = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_save += 1

        if self.epochs_since_save >= self.save_freq:
            metric = logs.get(self.monitor)
            save = True
            if self.save_best_only:
                if (self.best is None) or (metric < self.best):
                    self.best = metric
                    save = True
                else:
                    save = False
            if save:
                self.model.save(self.save_path)
            self.epochs_since_save = 0
