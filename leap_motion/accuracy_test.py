import sys
import unittest

from lib.test.argument_parser import wrapper
from lib.test import test_utils


if __name__ == '__main__':
    # Parse all args and pass on the args for unittest
    wrapper.parse_args()

    unittest.main(module=None, argv=sys.argv, exit=False)

    test_utils.plot_stats()
