"""Compatibility entry point for the original floodnet experiment."""
import sys

from disasterkd.cli import main


if __name__ == '__main__':
    main(['mask', '--dataset', 'floodnet', '--method', 'maskd', *sys.argv[1:]], legacy=True)
