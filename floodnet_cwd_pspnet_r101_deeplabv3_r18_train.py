"""Compatibility entry point for the original floodnet experiment."""
import sys

from disasterkd.cli import main


if __name__ == '__main__':
    main(['distill', '--dataset', 'floodnet', '--method', 'cwd',
          '--student', 'deeplabv3-18', *sys.argv[1:]], legacy=True)
