import matplotlib.pyplot as plt
import argparse
import numpy as np

# Plot the number of verified/unverified and time for verification at each epsilon value
def plotSweepingEPS(epss: list, num_verified: list, num_unverified: list, verifiedTime: list):
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(epss, num_verified, label='Verified')
    ax[0].plot(epss, num_unverified, label='Verified')
    ax[1].plot(epss, verifiedTime)

    ax[0].set_xlabel('$\epsilon$')
    ax[1].set_xlabel('$\epsilon$')
    ax[0].set_ylabel('Number of Instances')
    ax[1].set_ylabel('Time (s)')

    ax[0].legend()
    plt.savefig('results/sweeping_eps.png')
    # plt.show()

def parseInstanceCsv(file):
    insts = []
    with open(file) as f:
        for line in f.readlines():
            l = line.split(',')
            insts.append(l[1])
    return insts

def parseABcrownLog(file):
    insts = []
    times = []
    with open(file) as f:
        for line in f.readlines():
            if 'Result' in line:
                times.append(line.split(' ')[-2])
                if 'unsafe-pgd' in line:
                    insts.append(0)
                elif 'safe' in line:
                    insts.append(1)
                elif 'unknown' in line:
                    insts.append(1)
    return (insts, times)


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--log", type=str, help="Path to log file to parse")
    ap.add_argument('-i', "--instances", type=str, help="Path to instances csv")
    args = ap.parse_args()

    insts = parseInstanceCsv(args.instances)
    data = parseABcrownLog(args.log)
    #print("INSTANCES")
    #print(insts)

    epss = [(e.split('_')[2])[:-7] for i, e in enumerate(insts) if i % 10 ==0]
    #print(epss)
    #print("DATA")
    #print(data)

    num_verified = [sum(data[0][i:i+10]) for i in range(0, len(data[0]), 10)]
    num_unverified = [(10-sum(data[0][i:i+10])) for i in range(0, len(data[0]), 10)]

    print(num_unverified)
    print(num_verified)

    ts = [float(data[1][i]) for i in range(len(data[1]))]
    timeseg = [ts[i:i+10] for i in range(0, len(ts), 10)]

    times = []
    for s in timeseg:
        filters = [t for t, i in zip(s, data[0]) if i==1]
        if filters:
            times.append(np.mean(filters))
    plotSweepingEPS(epss, num_verified, num_unverified, times)