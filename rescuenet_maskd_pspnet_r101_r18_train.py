"""Compatibility entry point for the original rescuenet experiment."""
import sys

from disasterkd.cli import main


if __name__ == '__main__':
    main(['distill', '--dataset', 'rescuenet', '--method', 'maskd',
          '--student', 'pspnet18', *sys.argv[1:]], legacy=True)
