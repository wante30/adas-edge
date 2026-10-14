"""Keyboard/mouse controls shared by native UI and scripted simulation tests."""
from dataclasses import replace

from competition_ai.scenario import Controls, Lane, Maneuver, Mode, Motion


class Controller:
    def __init__(self):
        self.controls = Controls()
        self.paused = False
        self.binding = False

    def key(self, key):
        if isinstance(key,int):
            if key == 27:
                return 'quit'
            if not 0 <= key < 128:
                return None
            key = chr(key)
        lanes = {'1':Lane.LEFT,'2':Lane.STRAIGHT,'3':Lane.RIGHT}
        motions = {'0':Motion.AUTO,'s':Motion.STOP,'g':Motion.GO}
        modes = {'i':Mode.INTERSECTION,'d':Mode.ROAD,'n':Mode.NAVIGATION,'h':Mode.HAZARD}
        requests = {'a':Maneuver.AUTO,'q':Maneuver.CHANGE_LEFT,'e':Maneuver.CHANGE_RIGHT,
                    'j':Maneuver.TURN_LEFT,'l':Maneuver.TURN_RIGHT}
        if key in lanes:
            self.controls = replace(self.controls,lane=lanes[key])
            self.binding = False
        elif key in motions:
            self.controls = replace(self.controls,motion=motions[key])
        elif key in modes:
            self.controls = replace(self.controls,mode=modes[key],request=Maneuver.AUTO)
        elif key in requests:
            self.controls = replace(self.controls,request=requests[key])
        elif key == ' ':
            self.paused = not self.paused
        elif key == '.':
            self.paused = True
            return 'step'
        elif key == 'b':
            self.binding = True
        elif key == 'r':
            self.__init__()
            return 'restart'
        elif key == 'o':
            return 'open'
        return None
