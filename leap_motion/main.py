import argparse
import sys

from lib.leap_motion import Leap
from lib.leap_motion.LeapMotionListener import LeapMotionListener
from lib.leap_motion.LeapMotionListenerCyberHand import LeapMotionListenerCyberHand
from lib.leap_motion.LeapMotionFloatingHandListener import LeapMotionListenerFloatingHand


def main(args):
    # Create a listener
    if args["float"]:
        listener = LeapMotionListenerFloatingHand(num_samples=args["samples"],
                                                  with_object=args["object"],
                                                  render=args["render"])
    elif args["cyber_hand"]:
        listener = LeapMotionListenerCyberHand(num_samples=args["samples"],
                                               with_object=args["object"],
                                               render=args["render"])
    else:
        listener = LeapMotionListener(num_samples=args["samples"],
                                      with_object=args["object"],
                                      render=args["render"])

    # Create a controller
    controller = Leap.Controller()

    # Connect listener and controller
    controller.add_listener(listener)

    # Simple loop
    print("Press Enter to quit...")
    try:
        sys.stdin.readline()
    except KeyboardInterrupt:
        pass
    finally:
        # Remove listener
        controller.remove_listener(listener)


if __name__ == "__main__":
    # Check for sample / test mode
    parser = argparse.ArgumentParser(description="Run LeapMotionListener with/without sampling/testing mode")
    parser.add_argument("-s", "--samples",
                        help="sampling mode with specified number of samples",
                        type=int,
                        required=False,
                        default=None)
    parser.add_argument("-o", "--object",
                        help="object mode",
                        action="store_true",
                        required=False)
    parser.add_argument("-r", "--render",
                        help="render mode, only supports gif",
                        action="store_true",
                        required=False)
    parser.add_argument("-ch", "--cyber-hand",
                        help="use cyber-hand",
                        action="store_true",
                        required=False)
    parser.add_argument("-f", "--float",
                        help="floating hand",
                        action="store_true",
                        required=False)
    args = vars(parser.parse_args())

    main(args)
