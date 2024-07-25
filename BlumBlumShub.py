class BBSrandom:
    def __init__(self, seed = 17):
        self.M = 7823 * 6691
        self.x = seed
    def next(self, m = 100):
        self.x = (self.x)**2 % self.M
        return self.x % m

