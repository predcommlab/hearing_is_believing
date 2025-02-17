import numpy as np

from dataclasses import dataclass, field
from typing import Any, Union
from trigger import Port

# import psychopy and set hardware specs
from psychopy import visual, misc, core, prefs
prefs.hardware['audioLib'] = ['ptb', 'pyo']
prefs.hardware['audioLatencyMode'] = 3
from psychopy import sound
from psychopy.hardware import keyboard
import psychtoolbox as ptb

# trial type definitions
TRIAL_TYPE_PRACTICE = 0
TRIAL_TYPE_CONTROL = 1
TRIAL_TYPE_MAIN = 2
TRIAL_TYPE_PL1 = 3
TRIAL_TYPE_PL2 = 4
TRIAL_TYPE_ODD_GOOD = 5
TRIAL_TYPE_ODD_BAD = 6
TRIAL_TYPE_SURVEY = 7

# trigger definitions
TRIGGER_TRIAL_START = '1'
TRIGGER_TRIAL_CUE = '2'
TRIGGER_TRIAL_AUDITORY = '3'
TRIGGER_TRIAL_AUDITORY_OFF = '4'
TRIGGER_TRIAL_RESPONSE = '5'
TRIGGER_TRIAL_RESPONSE_POST = '6'

@dataclass
class Trial:
    context: str = field(default_factory = lambda: None)
    speaker: str = field(default_factory = lambda: None)
    preload_speaker: visual.ImageStim = field(default_factory = lambda: None)
    stimulus: str = field(default_factory = lambda: None)
    preload_stimulus: sound.Sound = field(default_factory = lambda: None)
    duration: float = field(default_factory = lambda: 0.)
    type: int = field(default_factory = lambda: TRIAL_TYPE_MAIN)
    no: int = field(default_factory = lambda: 0)
    block: int = field(default_factory = lambda: 0)
    options: tuple[str, str] = field(default_factory = lambda: [None, None])
    target_position: str = field(default_factory = lambda: None)
    rt: float = field(default_factory = lambda: None)
    choice: str = field(default_factory = lambda: None)
    correct: str = field(default_factory = lambda: None)
    ts: float = field(default_factory = lambda: 0.)
    t0: float = field(default_factory = lambda: 0.)
    t1: float = field(default_factory = lambda: 0.)
    t2: float = field(default_factory = lambda: 0.)
    t3: float = field(default_factory = lambda: 0.)
    t4: float = field(default_factory = lambda: 0.)
    t5: float = field(default_factory = lambda: 0.)

def instruction(slides: list[str], win: visual.Window):
    '''
    '''

    i = 0
    while i < len(slides):
        instructions = visual.TextStim(win, pos = [0, 0], text = slides[i], units = 'deg', height = 0.85, wrapWidth = 35)
        instructions.draw()
        win.flip()
        choice, rt = get_response(['left', 'right'], max_rt = None)
        i += 1 if choice == 'right' else -1
        i = np.clip(i, 0, len(slides))

def loading(win: visual.Window):
    '''
    '''

    status = visual.TextStim(win, pos = [0, 0], text = 'Aufgaben werden geladen...')
    status.draw()
    win.flip()

def countdown(win: visual.Window, L: float = 5.0, fr: float = 0.1, d: float = 0.25):
    '''
    '''

    # setup clocks
    rc = core.Clock()
    rc.reset()

    # setup timing
    cont = rc.getTime() < L
    last_r = -1
    
    # loop until max_rt is reached
    while cont:
        # remaining time
        r = np.ceil(L - rc.getTime()).astype(int)

        # flip only when necessary
        if last_r != r:
            last_r = r

            if r > 0:
                count = visual.TextStim(win, pos = [0, 0], text = f'{r}', color = [-1, -1, -1], colorSpace = 'rgb')
                count.draw()
            
            win.flip()

        # break loop prematurely (if necessary)
        cont = rc.getTime() < L

        # wait briefly before next update
        core.wait(fr)
    
    # delay
    core.wait(d)

def fixation(win: visual.Window, trigger: Union[Port, None] = None, t: float = 0.5, d: float = 0.050, flip: bool = True):
    '''
    '''

    # present cross
    cross = visual.TextStim(win, pos = [0, 0], text = '+', color = [-1, -1, -1], colorSpace = 'rgb')
    cross.draw()
    if trigger is not None: win.callOnFlip(trigger.send, TRIGGER_TRIAL_START)
    win.flip()

    # wait and flip
    core.wait(t)
    if flip: win.flip()
    core.wait(d)

def cue(image: visual.ImageStim, win: visual.Window, trigger: Port, T: str = TRIGGER_TRIAL_CUE, t: float = 0.75, d: float = 0.0, persist: bool = True, options: Union[list[str], None] = None, opt_colors: Union[list[list[float]], None] = None):
    '''
    '''

    # present image
    image.draw()

    # if desired, present options
    if options is not None:
        # draw box for contrast
        box = visual.Rect(win, 8, 2.5, 'deg', pos = [0, 0], fillColor = [1,1,1], fillColorSpace = 'rgb', opacity = 0.35, depth = 1)

        # setup options
        if opt_colors is None: opt_colors = [[-1,-1,-1], [-1,-1,-1]]

        if opt_colors is not None and len(opt_colors) > 0: opt_down = visual.TextStim(win, pos = [0, -0.75], text = options[0], color = opt_colors[0], depth = 2, units = 'deg', height = 0.85)
        else: opt_down = visual.TextStim(win, pos = [0, -0.75], text = options[0], depth = 2, units = 'deg', height = 0.85)

        if opt_colors is not None and len(opt_colors) > 1: opt_up = visual.TextStim(win, pos = [0, 0.75], text = options[1], color = opt_colors[1], depth = 2, units = 'deg', height = 0.85)
        else: opt_up = visual.TextStim(win, pos = [0, 0.75], text = options[1], depth = 2, units = 'deg', height = 0.85)
        
        # draw
        box.draw()
        opt_down.draw()
        opt_up.draw()
    
    # keep fixation
    cross = visual.TextStim(win, pos = [0, 0], text = '+', color = [-1, -1, -1], colorSpace = 'rgb')
    cross.draw()

    # flip
    win.callOnFlip(trigger.send, T)
    win.flip()

    # wait and flip
    core.wait(t)
    if not persist: win.flip()
    core.wait(d)

def get_response(valid: list[str] = [], max_rt: Union[float, None] = 2.0, allow_exit: bool = True) -> tuple[str, float]:
    '''
    '''

    # setup clocks
    rc = core.Clock()
    rc.reset()
    kb = keyboard.Keyboard()
    kb.clearEvents()
    kb.clock.reset()

    # setup response variables
    choice = None
    rt = None
    cont = rc.getTime() < max_rt if max_rt is not None else True
    
    # loop until max_rt is reached
    while cont:
        # get events
        keys = kb.getKeys()
        
        # handle events
        if len(keys) > 0:
            for key in keys:
                # accept only valid keys (or escape if allowed)
                if key.name in valid: 
                    choice = key.name
                    rt = key.rt
                    break
                elif key.name == 'escape' and allow_exit:
                    core.quit()
                    break
            
            # unclog cue
            kb.clearEvents()
        
        # break loop prematurely (if necessary)
        cont = rc.getTime() < max_rt if max_rt is not None else True
        if choice is not None: break
    
    return choice, rt

def play_sound(audio: sound.Sound, win: visual.Window, trigger: Port, d: float = 0.025, constant: bool = True, L: float = 1.5, persist: bool = True, flip: bool = True) -> float:
    '''
    '''

    # play sound at flip
    if flip:
        nflip = win.getFutureFlipTime(clock = 'ptb')
        audio.play(when = nflip)
        while ptb.GetSecs() < nflip: continue
        trigger.send(TRIGGER_TRIAL_AUDITORY)
        #win.callOnFlip(trigger.send, TRIGGER_TRIAL_AUDITORY)
    else:
        audio.play()
        trigger.send(TRIGGER_TRIAL_AUDITORY)

    # wait for sound/constant length and/or delay
    if persist and not constant: core.wait(len(audio.sndFile) / audio.sampleRate + d)
    elif not persist and not constant: core.wait(d)
    elif not persist and constant: core.wait(d)
    elif persist and not constant: core.wait(L + d)

    return len(audio.sndFile) / audio.sampleRate

def tafc_audio_visual(T: Trial, win: visual.Window, trigger: Port, feedback: bool = True) -> Trial:
    '''
    '''

    # setup trial clock
    tc = core.Clock()
    t0 = tc.getTime()

    # fixation
    t1s = tc.getTime()
    fixation(win, trigger)

    # speaker cue
    t2s = tc.getTime()
    cue(T.preload_speaker, win, trigger, persist = True)

    # auditory stimulus
    t3s = tc.getTime()
    duration = play_sound(T.preload_stimulus, win, trigger, d = 0.40, constant = False, persist = True)
    trigger.send(TRIGGER_TRIAL_AUDITORY_OFF)

    # option presentation
    target, alternative = T.options
    if T.target_position == 'up': down, up = alternative, target
    else: down, up = target, alternative
    t4s = tc.getTime()
    cue(T.preload_speaker, win, trigger, T = TRIGGER_TRIAL_RESPONSE, t = 0.0, persist = True, options = [down, up])
    choice, rt = get_response(valid = ['down', 'up'], allow_exit = True)

    # feedback
    if feedback:
        opt_colors = [[-1,0,-1], [-1,-1,-1]] if T.target_position == 'down' else [[-1,-1,-1], [-1,0,-1]]
        if choice == 'down' and T.target_position == 'up': opt_colors = [[0.5,-1,-1], [-1,-1,-1]]
        elif choice == 'up' and T.target_position == 'down': opt_colors = [[-1,-1,-1], [0.5,-1,-1]]
        t5s = tc.getTime()
        cue(T.preload_speaker, win, trigger, T = TRIGGER_TRIAL_RESPONSE_POST, t = 0.75, d = np.random.uniform(low = 1.0, high = 1.4), persist = False, options = [down, up], opt_colors = opt_colors)
        win.flip()
    else:
        win.flip()
        t5s = tc.getTime()
        core.wait(np.random.uniform(low = 1.0, high = 1.4))

    # set timing
    T.t0 = t0
    T.t1 = t1s - t0
    T.t2 = t2s - t0
    T.t3 = t3s - t0
    T.t4 = t4s - t0
    T.t5 = t5s - t0

    # set duration
    T.duration = duration

    # set states
    T.choice = choice
    T.rt = rt
    T.correct = T.target_position == T.choice

    return T

def natural_listening(T: Trial, win: visual.Window, trigger: Port, ITI: float = 0.40, persist: bool = False) -> Trial:
    '''
    '''

    # setup trial clock
    tc = core.Clock()
    t0 = tc.getTime()

    # fixation
    t1s = tc.getTime()
    fixation(win, trigger, flip = False)

    # play audio
    t2s = tc.getTime()
    D = play_sound(T.preload_stimulus, win, trigger, constant = False, persist = False, d = 0.0, flip = True)
    t2e = tc.getTime()
    fixation(win, None, t = 0.0, d = 0.0, flip = False)

    # loop to enable exit (if desired)
    while tc.getTime() < (t2s + D):
        get_response(valid = [], max_rt = 0.5, allow_exit = True)

    # final time stamp
    t3s = tc.getTime()
    trigger.send(TRIGGER_TRIAL_AUDITORY_OFF)

    # reset
    if not persist:
        win.flip()
        core.wait(ITI)

    # set timing
    T.t0 = t0
    T.t1 = t1s - t0
    T.t2 = t2s - t0
    T.t3 = t2e - t0
    T.t4 = t3s - t0

    # set duration
    T.duration = D

    return T

def survey(T: Trial, opts: list[tuple[visual.TextStim, str]], win: visual.Window) -> Trial:
    '''
    '''

    # setup trial clock
    tc = core.Clock()
    t0 = tc.getTime()

    # fixation
    t1s = tc.getTime()
    cross = visual.TextStim(win, pos = [0, 0], text = '+', color = [-1, -1, -1], colorSpace = 'rgb')
    cross.draw()
    win.flip()
    core.wait(0.25)

    # create and draw question
    t2s = tc.getTime()
    question = visual.TextStim(win, pos = [0, 0], text = T.stimulus, color = [-1, -1, -1], colorSpace = 'rgb', units = 'deg', wrapWidth = 30)
    question.draw()

    # create and draw options
    t3s = tc.getTime()
    valid = []
    for opt in opts:
        box, key = opt
        box.draw()
        valid.append(key)
    win.flip()

    # get response
    t4s = tc.getTime()
    choice, rt = get_response(valid = valid, max_rt = None)

    # set data
    t5s = tc.getTime()
    T.rt = rt
    T.choice = choice

    # set timing
    T.t0 = t0
    T.t1 = t1s - t0
    T.t2 = t2s - t0
    T.t3 = t3s - t0
    T.t4 = t4s - t0
    T.t5 = t5s - t0

    return T
