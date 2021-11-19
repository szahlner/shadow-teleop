import sys
import argparse


class UnitTestParser(object):
    """
    Class for parsing arguments.
    """

    def __init__(self):
        self.args = None

    def parse_args(self):
        # Parse optional extra arguments
        parser = argparse.ArgumentParser(
            description="Pass in the standard unittest arguments and optionally the accuracy")
        parser.add_argument("-a", "--accuracy",
                            help="accuracy to be used",
                            type=int,
                            default=1)
        ns, args = parser.parse_known_args()
        self.args = vars(ns)

        # Set the sys.argv to the unittest_args (leaving sys.argv[0] alone)
        sys.argv[1:] = args


wrapper = UnitTestParser()
