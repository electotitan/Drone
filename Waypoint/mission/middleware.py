"""
Tiny in-process stand-in for the ROS 2 pieces this task needs -- NO ROS 2 involved.

    ServiceServer / ServiceClient   request -> response, by name
    ActionServer  / ActionClient    goal -> (feedback ...) -> result, by name

Each server runs in its own thread, clients talk to servers only through
queues/futures (never by calling server methods directly), and everything is
looked up by *name* in a registry -- the same shape as rclpy, so porting this
to real ROS 2 later means swapping this file for rclpy and keeping the nodes.

Action semantics (same as ROS 2):
    client.send_goal_async(goal, feedback_callback)  -> Future[ClientGoalHandle]
    handle.accepted                                  -> bool
    handle.get_result_async()                        -> Future[(status, result)]
    handle.cancel_goal_async()
    server execute callback(goal_handle) -> result; it calls goal_handle.publish_feedback(fb)
    and goal_handle.succeed() / abort() / canceled() before returning.
This server accepts ONE goal at a time and rejects others while busy.
"""
import queue
import threading
import time

# --------------------------------------------------------------------------- futures
class Future:
    def __init__(self):
        self._ev = threading.Event()
        self._value = None

    def set_result(self, value):
        self._value = value
        self._ev.set()

    def done(self):
        return self._ev.is_set()

    def result(self, timeout=None):
        if not self._ev.wait(timeout):
            raise TimeoutError("future not completed in time")
        return self._value


_registry_lock = threading.Lock()
_services = {}
_actions = {}


# --------------------------------------------------------------------------- services
class ServiceServer:
    def __init__(self, name, handler):
        self.name, self._handler = name, handler
        self._q = queue.Queue()
        self._stop = threading.Event()
        with _registry_lock:
            _services[name] = self
        self._thread = threading.Thread(target=self._spin, name=f"srv:{name}", daemon=True)
        self._thread.start()

    def _spin(self):
        while not self._stop.is_set():
            try:
                request, fut = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            fut.set_result(self._handler(request))

    def shutdown(self):
        self._stop.set()
        with _registry_lock:
            _services.pop(self.name, None)


class ServiceClient:
    def __init__(self, name):
        self.name = name

    def wait_for_service(self, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end:
            with _registry_lock:
                if self.name in _services:
                    return True
            time.sleep(0.02)
        return False

    def call_async(self, request):
        fut = Future()
        with _registry_lock:
            server = _services.get(self.name)
        if server is None:
            raise RuntimeError(f"service '{self.name}' is not available")
        server._q.put((request, fut))
        return fut

    def call(self, request, timeout=5.0):
        return self.call_async(request).result(timeout)


# --------------------------------------------------------------------------- actions
class GoalStatus:
    ACCEPTED = "ACCEPTED"
    EXECUTING = "EXECUTING"
    SUCCEEDED = "SUCCEEDED"
    ABORTED = "ABORTED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


class ClientGoalHandle:
    def __init__(self, goal, feedback_cb):
        self.goal = goal
        self.accepted = False
        self.status = GoalStatus.REJECTED
        self._feedback_cb = feedback_cb
        self._result_fut = Future()
        self._cancel_flag = threading.Event()

    def get_result_async(self):
        return self._result_fut

    def cancel_goal_async(self):
        self._cancel_flag.set()


class ServerGoalHandle:
    def __init__(self, client_handle):
        self._ch = client_handle
        self.request = client_handle.goal
        self.status = GoalStatus.EXECUTING
        self._ch.status = GoalStatus.EXECUTING

    @property
    def is_cancel_requested(self):
        return self._ch._cancel_flag.is_set()

    def publish_feedback(self, feedback):
        if self._ch._feedback_cb is not None:
            self._ch._feedback_cb(feedback)

    def succeed(self):
        self.status = GoalStatus.SUCCEEDED

    def abort(self):
        self.status = GoalStatus.ABORTED

    def canceled(self):
        self.status = GoalStatus.CANCELED


class ActionServer:
    def __init__(self, name, execute_callback, goal_callback=None):
        self.name = name
        self._execute = execute_callback
        self._goal_cb = goal_callback or (lambda goal: True)
        self._q = queue.Queue()
        self._busy = threading.Event()
        self._stop = threading.Event()
        with _registry_lock:
            _actions[name] = self
        self._thread = threading.Thread(target=self._spin, name=f"act:{name}", daemon=True)
        self._thread.start()

    def _spin(self):
        while not self._stop.is_set():
            try:
                ch, accept_fut = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            if self._busy.is_set() or not self._goal_cb(ch.goal):
                ch.accepted, ch.status = False, GoalStatus.REJECTED
                ch._result_fut.set_result((GoalStatus.REJECTED, None))
                accept_fut.set_result(ch)
                continue
            self._busy.set()
            ch.accepted, ch.status = True, GoalStatus.ACCEPTED
            accept_fut.set_result(ch)
            threading.Thread(target=self._run_goal, args=(ch,), daemon=True,
                             name=f"act:{self.name}:goal").start()

    def _run_goal(self, ch):
        gh = ServerGoalHandle(ch)
        try:
            result = self._execute(gh)
        except Exception as exc:                    # never leave the client hanging
            gh.abort()
            result = None
            print(f"[{self.name}] execute callback raised: {exc!r}")
        ch.status = gh.status
        self._busy.clear()
        ch._result_fut.set_result((gh.status, result))

    def shutdown(self):
        self._stop.set()
        with _registry_lock:
            _actions.pop(self.name, None)


class ActionClient:
    def __init__(self, name):
        self.name = name

    def wait_for_server(self, timeout=10.0):
        end = time.time() + timeout
        while time.time() < end:
            with _registry_lock:
                if self.name in _actions:
                    return True
            time.sleep(0.02)
        return False

    def send_goal_async(self, goal, feedback_callback=None):
        with _registry_lock:
            server = _actions.get(self.name)
        if server is None:
            raise RuntimeError(f"action server '{self.name}' is not available")
        ch = ClientGoalHandle(goal, feedback_callback)
        fut = Future()
        server._q.put((ch, fut))
        return fut
