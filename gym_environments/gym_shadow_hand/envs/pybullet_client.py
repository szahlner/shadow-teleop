import functools
import inspect
import pybullet


class PyBulletClient(object):

    def __init__(self, connection_mode=None, render_options=None):
        """Creates a PyBullet client and connects to a simulation."""
        if connection_mode is None:
            self.client = pybullet.connect(pybullet.SHARED_MEMORY)

            if self.client >= 0:
                return
            else:
                connection_mode = pybullet.DIRECT

        self.client = pybullet.connect(connection_mode)

    def __del__(self):
        """Clean up connection if not already done."""
        if self.client >= 0:
            try:
                pybullet.disconnect(physicsClientId=self.client)
                self.client = -1
            except pybullet.error:
                pass

    def __getattr__(self, name):
        """Inject the client id into Bullet functions."""
        attribute = getattr(pybullet, name)

        if inspect.isbuiltin(attribute):
            attribute = functools.partial(attribute, physicsClientId=self.client)
        if name == "disconnect":
            self.client = -1

        return attribute