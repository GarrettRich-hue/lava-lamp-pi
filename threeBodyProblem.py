import timeit
import itertools
import math
import cv2 as cv
import numpy as np
class ThreeBody:
    def __iter__(self, speed = 0.005, G = 0.005, dt = 0.01, delta_t = 0.05):
        self.res = (40,40) 
        self.N = 3
        self.r = [np.random.rand(2) for i in range(self.N)]
        self.v = [np.random.rand(2)*speed for i in range(self.N)]
        self.col = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        self.col = [np.array(i, dtype=np.uint8) for i in self.col]
        self.m = [1.0, 1.5, 2.0]
        self.G = G
        self.dt = dt
        self.delta_t = delta_t
        self.nt = int(math.floor(self.delta_t/self.dt))
        self.temp = np.zeros(2)
        self.acctemp = np.zeros(2)
        return self
    def updatePhysics(self):
        for i in range(self.nt):
            self.step()
    def step(self):

        a = [np.zeros(2, dtype=float) for i in range(self.N)]
        temp = self.temp
        for i in range(self.N-1):
            for j in range(i+1, self.N):
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        temp[0] = x
                        temp[1] = y
                        temp += self.r[j]
                        temp -= self.r[i]
                        rad = np.linalg.norm(temp)
                        if rad < 0.1:
                            continue
                        temp *= self.G * self.m[i] * self.m[j]  / rad**3
                        a[i] += temp
                        a[j] -= temp
        acctemp = self.acctemp
        for i in range(self.N):
            temp[:] = self.v[i]
            temp *= self.dt

            acctemp[:] = a[i]
            acctemp *= self.dt
            self.v[i] += acctemp
            acctemp *= 0.5 * self.dt
            self.r[i] += temp
            self.r[i] += acctemp
            self.r[i] %= 1
    def plot(self, im, pos, col):
        coor = pos * np.array(self.res)
        ic = np.floor(coor)
        fc = coor - ic
        #get the percentage of line that is within the cell the centre is in
        intoLine = np.vectorize(lambda x: abs(x-0.5)+0.5)
        o = intoLine(fc)
        #calculate percentage that is escaping out of cell
        es = np.array([1., 1.]) - o
        #calculate the offsets to the neighbouring cells that the circle is clipping into
        intoCell = np.vectorize(lambda x: 1 if x>0.5 else -1)
        c = intoCell(fc)
        
        #calculate the area of the escaping segment
        area = es*np.flip(o)
        #calculate the area of the diagonal escape
        diag = np.prod(es)
        #calculate area within cell
        within = np.prod(o)  
        #calculate percentages
        total = 1
        withinP = within/total
        diagP = diag/total
        areaP = area/total

        X, Y = int(ic[0]), int(ic[1])
        nX, nY = (X+c[0])%self.res[0], (Y+c[1])%self.res[1]
        im[X, Y] += (withinP*col).astype(np.uint8)
        im[nX, Y] += (areaP[0]*col).astype(np.uint8)
        im[X, nY] += (areaP[1]*col).astype(np.uint8)
        im[nX, nY] += (diagP*col).astype(np.uint8)
    def genImage(self):
        im = np.zeros(self.res + (3,), dtype = np.uint8)
        for i in range(self.N):
            self.plot(im, self.r[i], self.col[i])
        return im
    def __next__(self):
        self.updatePhysics()
        return self.genImage()
if __name__ == "__main__":
    print("Benchmark simulation")
    print(f'{timeit.timeit(lambda: [i for i in itertools.islice(iter(ThreeBody()),60)], number = 10)}s to create 60 frames of video 10 times')
    print("Generating video")
    videoRes = (320, 320)
    sim = iter(ThreeBody())
    result = cv.VideoWriter("threeBody.mp4", cv.VideoWriter_fourcc(*'MP4V'), 30, videoRes)
    for i in range(30*15):
        frame = next(sim)
        bigger = cv.resize(frame, videoRes, interpolation = cv.INTER_NEAREST)
        result.write(bigger)
        if i % 30 == 0:
            print(f'Completed {i/30}s')
    print("wrote everything")
    result.release()
