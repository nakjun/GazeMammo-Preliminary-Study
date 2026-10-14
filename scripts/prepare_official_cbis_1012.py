#!/usr/bin/env python
"""Audit official metadata, then explicitly convert development pixels."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.official_cbis_data_1012 import RAW_ROOT, RUN_ROOT, build_cohort, convert_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['cohort', 'convert'])
    parser.add_argument('--output', type=Path, default=RUN_ROOT)
    parser.add_argument('--metadata-dir', type=Path)
    parser.add_argument('--raw-root', type=Path, default=RAW_ROOT)
    parser.add_argument('--split', choices=['trainval', 'test'], default='trainval')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.stage == 'cohort':
        build_cohort(args.metadata_dir or args.output / 'metadata', args.raw_root, args.output)
        print((args.output / 'audit.json').read_text())
    else:
        state = convert_manifest(args.output, args.split, args.limit)
        print(json.dumps(dict(split=state['split'], complete=state['complete'],
                              converted_rows=len(state['row_hashes']))))


if __name__ == '__main__':
    main()
