import os
from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Clean divorce-exp cache artifacts")
    parser.add_argument('--dry-run', action='store_true', help='Show files to delete without deleting')
    args = parser.parse_args()

    root = Path(__file__).parent
    patterns = [
        'divorce_simulations*.json',
        'evaluation_results*.json',
        'divorce_evaluation_results*.json',
        'divorce_simulations_sample.txt',
        'divorce_personas_sample.txt'
    ]

    to_delete = []
    for pat in patterns:
        to_delete.extend(root.glob(pat))

    if not to_delete:
        print('No cache files found.')
        return

    print('Files to delete:')
    for p in to_delete:
        print(' -', p.name)

    if args.dry_run:
        print('\nDry run: no files deleted')
        return

    for p in to_delete:
        try:
            p.unlink()
        except Exception as e:
            print(f'Failed to delete {p}: {e}')

    print('\n✅ Cache cleared.')


if __name__ == '__main__':
    main()
