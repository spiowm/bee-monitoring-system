#!/usr/bin/env python3
import os
from collections import defaultdict
from pathlib import Path

def main():
    labels_dir = Path("/home/spiowm/spi/projects/bee-monitoring-system/research/datasets/raw/pose/labels")
    hives = ['20230609a', '20230609b', '20230609c', '20230609d', '20230609e', '20230711a', '20230711b', '20230711c']
    
    counts = defaultdict(int)
    
    if not labels_dir.exists():
        print(f"Directory not found: {labels_dir}")
        return

    for filepath in labels_dir.glob("*.txt"):
        filename = filepath.name
        
        hive_id = None
        for h in hives:
            if filename.startswith(h):
                hive_id = h
                break
                
        if hive_id:
            try:
                with open(filepath, 'r') as f:
                    lines = f.readlines()
                    num_bees = len([line for line in lines if line.strip()])
                    counts[hive_id] += num_bees
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    # Print the table
    print(f"{'Вулик':<15} | {'Кількість бджіл':<15}")
    print("-" * 35)
    
    total_bees = 0
    for hive in hives:
        count = counts[hive]
        total_bees += count
        print(f"{hive:<15} | {count:<15}")
        
    print("-" * 35)
    print(f"{'Загалом':<15} | {total_bees:<15}")

if __name__ == "__main__":
    main()
