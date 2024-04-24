import matplotlib.pyplot as plt
import argparse
import numpy as np
import os

NUM_INSTANCES = 50
BENCHMARK = "cifar"

# Plot the number of verified/unverified and time for verification at each epsilon value
def plotSweepingEPS(epss: list, num_verified: list, num_unverified: list, verifiedTime: list, verifier):
    fig, ax = plt.subplots(figsize=(10,5))
    axx = ax.twinx()
    ax.plot(epss, num_verified, label='Verified')
    ax.plot(epss, num_unverified, label='Unverified')
    axx.plot(epss, verifiedTime, label='Time', color='k')

    ax.set_xlabel('$\epsilon$')
    ax.set_xlabel('$\epsilon$')
    ax.set_ylabel('Number of Instances (per $\epsilon$)')
    axx.set_ylabel('Time (s)')

    ax.legend()
    axx.legend()
    plt.title(f'Sweeping $\epsilon$ for {BENCHMARK.capitalize()} (MNIST)')
    plt.xticks(np.arange(0, 0.025, .002))
    plt.savefig(f'results/sweeping_eps_{BENCHMARK}_{NUM_INSTANCES}_{verifier}.png')
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
                    insts.append(0)
    num_verified = [sum(insts[i:i+NUM_INSTANCES]) for i in range(0, len(insts), NUM_INSTANCES)]
    num_unverified = [(NUM_INSTANCES-sum(insts[i:i+NUM_INSTANCES])) for i in range(0, len(insts), NUM_INSTANCES)]

    ts = [float(times[i]) for i in range(len(times))]
    timeseg = [ts[i:i+NUM_INSTANCES] for i in range(0, len(ts), NUM_INSTANCES)]

    times = []
    for s in timeseg:
        filters = [t for t, i in zip(s, insts) if i==1]
        if filters:
            times.append(np.mean(filters))
    return (num_verified, num_unverified, times)

def parseNeuralSATlog(file):
    data = {} # string -> list [int unverified, int verified, list [times]]

    with open(file) as f:
        for line in f.readlines():
            result = line.split(",")
            eps = result[1].split('_')[2][:-7]
            if eps not in data:
                data[eps] = [0, 0, []]
            if result[2] == 'unsat': # verified
                data[eps][1] += 1
            else: # unverified 
            # if result[2] == 'sat' or result[2] == 'timeout': # unverified
                data[eps][0] += 1
            
            data[eps][2].append(result[3].replace("\n", ''))
    
    sorted_data = dict(sorted(data.items()))
    epss = [] # epsilon values
    num_verified = [] # number of verified instances at an epsilon
    num_unverified = [] # number of unverified instances at an epsilon
    avg_time = [] # avg time to verify of all instances at an epsilon

    for eps in data:
        epss.append(float(eps))
        num_unverified.append(data[eps][0])
        num_verified.append(data[eps][1])
        times =  [float(i) for i in data[eps][2]]
        avg_time.append(sum(times) / len(times))

    return epss, num_verified, num_unverified, avg_time       



if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--log", type=str, help="Path to log file to parse")
    ap.add_argument('-i', "--instances", type=str, help="Path to instances csv")
    ap.add_argument('-v', "--verifier", type=str, help="'abc' or 'ns'")
    args = ap.parse_args()

    if args.verifier == 'ns':
        epss, num_verified, num_unverified, avg_time = parseNeuralSATlog(args.log)
        plotSweepingEPS(epss, num_verified, num_unverified, avg_time, args.verifier)

    if args.verifier == 'abc':
        insts = parseInstanceCsv(args.instances)
        epss = [float((e.split('_')[2])[:-7]) for i, e in enumerate(insts) if i % NUM_INSTANCES ==0]
        num_verified, num_unverified, times = parseABcrownLog(args.log)
        
        plotSweepingEPS(epss, num_verified, num_unverified, times, args.verifier)