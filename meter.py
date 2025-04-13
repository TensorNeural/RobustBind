class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    @property
    def avg(self):
        return (self.sum / self.count) if self.count else 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n