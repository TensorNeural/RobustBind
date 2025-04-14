import argparse
import matplotlib.pyplot as plt
import json

def plot_lr_finder(lrs, losses):
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("LR Finder")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learning Rate Viewer")
    parser.add_argument("--finder_result_file", type=str, default="./lr_finder_results.json", help="Path to the LR finder results file")
    
    with open(parser.finder_result_file, 'rb') as f:
        json_data = f.read()
        data = json.loads(json_data)
        lrs = data['lrs']
        losses = data['losses']
        plot_lr_finder(lrs, losses)