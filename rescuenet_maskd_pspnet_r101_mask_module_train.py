"""Compatibility entry point for the original rescuenet experiment."""
import sys

from disasterkd.cli import main


if __name__ == '__main__':
    main(['mask', '--dataset', 'rescuenet', '--method', 'maskd', *sys.argv[1:]], legacy=True)
