import os
import sys

if __name__ == "__main__":
    _TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
    if _TRAINING_DIR not in sys.path:
        sys.path.insert(0, _TRAINING_DIR)

    from common.run import build_parser, train_scout

    args = build_parser().parse_args()
    train_scout(args)
