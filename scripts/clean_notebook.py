#!/usr/bin/env python3
"""Strip private/transient notebook outputs in-place, creating .ipynb.bak backups."""
import argparse
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('notebooks', nargs='+')
    p.add_argument('--write', action='store_true', help='Modify notebooks; backup original first')
    a = p.parse_args()
    import nbformat
    for name in a.notebooks:
        path = Path(name)
        nb = nbformat.read(path, as_version=4)
        count = 0
        for cell in nb.cells:
            if cell.cell_type == 'code':
                count += len(cell.get('outputs', []))
                cell.outputs = []
                cell.execution_count = None
        nb.metadata.pop('widgets', None)
        print(path, 'outputs to remove:', count)
        if a.write:
            backup = path.with_name(path.name + '.bak')
            shutil.copy2(path, backup)
            nbformat.write(nb, path)
            print('Cleaned:', path, '; backup:', backup)


if __name__ == '__main__':
    main()
