"""
CLI interface; for detailed usage description, use the command `python . -h`
"""

from argparse import ArgumentParser
from main import process_stream

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input', type=str,
                        help='Input source; can be a file or a camera ID (for example: 0)')
    parser.add_argument('-x', '--input_resolution', type=int, nargs=2,
                        help='If specifid, the input stream will be clipped to the specified resolution')
    parser.add_argument('-o', '--output', type=str,
                        help='Output file; if unspecified, the output will be displayed in a window')
    parser.add_argument('-b', '--background', type=str,
                        help='The to-be-composed background in the output stream')
    parser.add_argument('-z', '--harmonization', choices=('model', 'color'),
                        help='Harmonization model to be used')
    parser.add_argument('-s', '--super_resolution', choices=(2, 3, 4, 8), type=int,
                        help='Super resolution scale')
    parser.add_argument('-p', '--postprocessing', type=float, nargs=2,
                        help='Postprocessing window size and intensity; helps with flickering')
    args = parser.parse_args()

    args.input = int(args.input) if args.input.isdigit() else args.input
    args.postprocessing = (int(args.postprocessing[0]), args.postprocessing[1]) if args.postprocessing else None

    process_stream(
        args.input, args.input_resolution, args.output,
        args.background, args.harmonization,
        args.super_resolution, args.postprocessing
    )
