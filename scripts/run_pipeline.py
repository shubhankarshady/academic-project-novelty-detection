#!/usr/bin/env python
"""
Run complete pipeline: data loading, feature extraction, section fusion training, and validation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train import main as train_main

def main():
    print("Executing Academic Project Novelty Detection Pipeline...")
    train_main()

if __name__ == "__main__":
    main()
