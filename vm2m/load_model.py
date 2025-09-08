# .th -> .bin
from utils import export
import argparse

def main():
    parser = argparse.ArgumentParser(description='Script for processing video and model paths.')
    
    parser.add_argument('--checkpoint_path', type=str, required=True, 
                        help='Path to the model checkpoint.')
    parser.add_argument('--output_path', type=str, required=True, 
                        help='Path to the output.')
    args = parser.parse_args()
    
    export.export_lm(args.checkpoint_path, args.output_path)

if __name__ == "__main__":
    main()
