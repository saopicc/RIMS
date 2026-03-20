# DynSpecMS/scripts/cli.py
import argparse
import sys

from DynSpecMS.scripts import ms2dynspec
from DynSpecMS.scripts import dynspec_upload

def main():
    parser = argparse.ArgumentParser(
        prog="rims",
        description="The Dynamic Spectra Extraction and Publishing Tool for Measurement Sets",
    )
    
    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="command",
        help="Choose a command to run"
    )
    
    subparsers.add_parser("run", help="Run the ms2dynspec extraction process", add_help=False)
    subparsers.add_parser("publish", help="Publish dynamic spectra to RIMS Online", add_help=False)
    # Future commands can easily be added here:

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args, remaining_argv = parser.parse_known_args()

    sys.argv = [f"rims {args.command}"] + remaining_argv

    if args.command == "run":
        ms2dynspec.main()
    elif args.command == "publish":
        dynspec_upload.main()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()