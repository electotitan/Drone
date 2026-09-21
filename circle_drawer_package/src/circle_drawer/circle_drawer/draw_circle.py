#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from turtlesim.srv import TeleportAbsolute, SetPen


class CircleDrawer(Node):
    """
    Draws a circle of diameter 2.0 centered at (5.0, 5.0) using turtlesim,
    then returns the turtle to the center point without drawing.
    """

    def __init__(self):
        super().__init__('circle_drawer')

        self.teleport_client = self.create_client(
            TeleportAbsolute, '/turtle1/teleport_absolute')
        self.pen_client = self.create_client(
            SetPen, '/turtle1/set_pen')

        self.get_logger().info('Waiting for turtlesim services...')
        while not self.teleport_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('  still waiting for /turtle1/teleport_absolute ...')
        while not self.pen_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('  still waiting for /turtle1/set_pen ...')
        self.get_logger().info('Services are ready.')

        # Circle parameters
        self.cx = 5.0
        self.cy = 5.0
        self.diameter = 2.0
        self.radius = self.diameter / 2.0

        self.draw_circle()

    # ---------- helper calls ----------

    def set_pen(self, r, g, b, width, off):
        req = SetPen.Request()
        req.r = r
        req.g = g
        req.b = b
        req.width = width
        req.off = off
        future = self.pen_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    def teleport(self, x, y, theta=0.0):
        req = TeleportAbsolute.Request()
        req.x = x
        req.y = y
        req.theta = theta
        future = self.teleport_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

    # ---------- main routine ----------

    def draw_circle(self):
        # 1) Lift the pen and jump to the starting point on the circle
        #    (rightmost point, angle = 0) so no stray line is drawn.
        self.set_pen(255, 255, 255, 3, 1)  # off = 1 -> pen up
        start_x = self.cx + self.radius
        start_y = self.cy
        self.teleport(start_x, start_y, math.pi / 2)

        # 2) Put the pen down and walk around the circle.
        self.set_pen(255, 255, 255, 3, 0)  # off = 0 -> pen down

        steps = 360
        for i in range(steps + 1):
            angle = 2.0 * math.pi * i / steps
            x = self.cx + self.radius * math.cos(angle)
            y = self.cy + self.radius * math.sin(angle)
            heading = angle + math.pi / 2  # face tangent to the circle
            self.teleport(x, y, heading)

        # 3) Lift the pen and return to the exact center (5.0, 5.0).
        self.set_pen(255, 255, 255, 3, 1)
        self.teleport(self.cx, self.cy, 0.0)

        self.get_logger().info(
            f'Done. Circle of diameter {self.diameter} drawn around '
            f'({self.cx}, {self.cy}); turtle parked at center.'
        )


def main(args=None):
    rclpy.init(args=args)
    node = CircleDrawer()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
