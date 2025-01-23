import os
import torch
import argparse

def combine_data(file_root, output_path):
    filenames = os.listdir(file_root)
    list_data = []
    data_all = {}
    for filename in filenames:
        file_path = os.path.join(file_root, filename)
        data = torch.load(file_path)
        data_all.update(data)      
    torch.save(data_all, output_path)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default='phi_sciqa.pt')
    parser.add_argument("--output_path", type=str, default='phi_sciqa.pt')
    args = parser.parse_args()

    combine_data(args.input_root, args.output_path)
