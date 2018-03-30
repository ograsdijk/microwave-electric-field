from potential_functions import potential_linecharge
import numpy as np
from scipy.optimize import brute
import multiprocessing as mp
import argparse
import time

def generate_wires():
    nwires = 4
    x0wire = .55
    y0wire = 2
    dxwire = 1.1
    dywire = 0
    wires = np.zeros((3, 4 * nwires))
    for n in range(nwires):
        wires[0][n] = -x0wire - n * dxwire
        wires[1][n] = y0wire + n * dywire
        wires[2][n] = 1
        wires[0][n + nwires] = x0wire + n * dxwire
        wires[1][n + nwires] = y0wire + n * dywire
        wires[2][n + nwires] = 1
        wires[0][n + nwires * 2] = -x0wire - n * dxwire
        wires[1][n + nwires * 2] = -y0wire - n * dywire
        wires[2][n + nwires * 2] = -1
        wires[0][n + nwires * 3] = x0wire + n * dxwire
        wires[1][n + nwires * 3] = -y0wire - n * dywire
        wires[2][n + nwires * 3] = -1
    return wires

def potential_wires(x, y, wires):
    xwires = wires[0]
    ywires = wires[1]
    charge = wires[2]
    potential = np.zeros(x.shape)
    for xw, yw, cw in zip(xwires, ywires, charge):
        potential += potential_linecharge(x, y, cw, xw, yw)

    potential[potential == np.inf] = np.nan
    potential[np.isnan(potential)] = np.nanmax(potential)
    potential[potential == -np.inf] = np.nan
    potential[np.isnan(potential)] = np.nanmin(potential)

    return potential

def minimize_function(wpos, x, y, wires):
    xc, yc = x.shape
    xc //= 2
    yc //= 2
    w = wires.copy()
    wpos = list(np.append(-wpos, wpos))*2
    w[0,:] = wpos
    potential = potential_wires(x, y, w)
    Ey, Ex = np.gradient(-potential)
    return ((Ey-Ey[xc,yc])**2).sum()/Ey[xc,yc]**2

def multi_fun(args):
    r, x, y, wires, procnr = args
    print(procnr, r)
    m = brute(minimize_function, args = (x, y, wires), ranges = r, finish = None)
    return m


if __name__ == "__main__":
    y,x = np.mgrid[1:-1:501j,1:-1:501j]
    wires = generate_wires()

    parser = argparse.ArgumentParser(description = "Brute Force Wire location Search, supply # cores")
    parser.add_argument('--nworkers', nargs = '?', help = '# multiprocessing cores', type = int, required = False)
    args = parser.parse_args()

    if args.nworkers == None:
        workers = 10

    rmin = 0.1
    rmax = 1
    rstep = 0.1

    ranges = [[(i + 1) * np.round(rmax / workers, 1), (i + 2) * np.round(rmax / workers, 1)] for i in range(workers)]
    ranges[0][0] = rmin
    r = [[slice(rmin, rmax+rstep, rstep)]*3]*workers
    for n in range(workers):
        if ranges[n][0] == ranges[n][1]:
            ranges[n][1] += rstep

        r[n] = [slice(rmin, rmax+rstep, rstep)]*3 + [slice(np.round(ranges[n][0],1), np.round(ranges[n][1],1), rstep)]

    arg = [(i, x, y, wires, idx) for idx, i in enumerate(r)]

    tstart = time.time()
    pool = mp.Pool(processes = workers)
    minimum = pool.map(multi_fun, arg)
    pool.close()
    pool.join()
    print(minimum)
    print('{0:.2f} s'.format(time.time()-tstart))