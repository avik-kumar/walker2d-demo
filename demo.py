from train_walker2d import *
from record_video import *

# TODO: Clean up code and comments.
# TODO: Fix video recording to not overwrite existing files.
# TODO: Add customization to reward policy (CustomRewardWrapper).
# TODO: Finalize project structure (demo.py file or not, etc).

def main():
    train_walker2d()
    record_video()

if __name__ == "__main__":
    main()