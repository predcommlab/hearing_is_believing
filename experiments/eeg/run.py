'''
File to run the EEG experimental sessions. In brief, this will
present the passive listening trials (with some attention checks, 
i.e. choice-based trials) of clear words, the passive listening
trial(s) of narrative listening plus comprehension questions,
then runs the main experiment (i.e., speaker-specific semantic
learning in 2AFC), and finally runs an "oddball" task (of sorts)
where we present excellent and terrible exemplars per speaker as
well as ask a couple final questions at the end.

Note that everything, including not only randomisation but also
the manipulation of the oddball task, is computed in real time
here. This also means, however, that a lot of things can be
customised using flags, if desired. If you would like to run
this task as we ran it for participants, please supply no such
additional flags. All defaults are set up such that the task
should reproduce our experiment precisely.

Note that, of course, this does not include the real-time
free-energy model and its manipulations. These are computed
stochastically, so results will vary not only by responses
but also from run to run.
'''

import numpy as np
import pandas as pd
import trial, trigger, data, model
import sys
import os

from psychopy import visual, core, prefs
prefs.hardware['audioLib'] = ['ptb', 'pyo']
prefs.hardware['audioLatencyMode'] = 3
from psychopy import sound

from typing import Union, Any

def get_opt(opt: str, default: Union[bool, str, int, float] = False) -> Union[bool, str, int, float]:
    '''
    Functions to obtain an argument from console inputs, with default values. Note that
    we omit dashes. For example, valid input would be:

        `python run.py fast=1 verbose=0 block_l=40`
    
    but _NOT_:

        `python run.py -fast=1 -verbose=0 -block_l=40`

    INPUTS:
        opt     -   Name of the flag to read out (if present).
        default -   Default value to supply if missing argument.
    
    OUTPUTS:
        arg     -   Argument or default.
    '''

    for k in sys.argv[1:]:
        if(k[0:len(opt)] == opt) and ((len(k) == len(opt)) or (k[len(opt)] == '=')): return k[len(opt)+1:]
    
    return default

if __name__ == '__main__':
    '''
    Preamble and flags
    '''

    # set flags
    __FLAG_BLOCK_LENGTH = int(get_opt('block_l', default = 20))
    __FLAG_ODDBALL_LENGTH = int(get_opt('oddbal_l', default = 20))
    __FLAG_ODDBALL_CATEGORY = int(__FLAG_ODDBALL_LENGTH / 2)
    __FLAG_MAX_BUFFER = int(get_opt('max_buffer', default = 3))
    __FLAG_MIN_REPETITIONS = int(get_opt('min_rep', default = 1))
    __FLAG_MAX_REPETITIONS = int(get_opt('max_rep', default = 5))
    __FLAG_PID_LENGTH = int(get_opt('pid_l', default = 6))
    __FLAG_FAST_MODE = bool(int(get_opt('fast', default = 0)))
    __FLAG_SKIP_AHEAD = int(get_opt('skip_ahead', default = 0))
    __FLAG_SKIP_AHEAD_PL1 = 1
    __FLAG_SKIP_AHEAD_PL2 = 2
    __FLAG_SKIP_AHEAD_PRACTICE = 3
    __FLAG_SKIP_AHEAD_MAIN = 4
    __FLAG_SKIP_AHEAD_ODDBALL = 5
    __FLAG_SKIP_AHEAD_SURVEY = 6
    __FLAG_DATA = str(get_opt('data', default = './data/'))
    __FLAG_TRIGGER_SIMULATE = bool(int(get_opt('trigger_sim', default = False)))
    __FLAG_TRIGGER_PORT = str(get_opt('trigger_port', default = '0xCFF8'))
    __FLAG_SCREEN = int(get_opt('screen', default = 1))
    __FLAG_VERBOSE = bool(int(get_opt('verbose', default = True)))
    __FLAG_PL_CHOICE_PROB = float(get_opt('pl_prob', default = 0.15))
    __FLAG_COUNTDOWN_L = float(get_opt('countdown', default = 5.0))

    '''
    Randomisation and loading of stimuli
    '''

    # setup psychopy and display loading screen
    global_clock = core.Clock()
    win = visual.Window(size = (1920, 1080), screen = __FLAG_SCREEN, monitor = "testMonitor", units = 'deg', fullscr = True)
    trial.loading(win)

    # setup passive listening task
    passive_listening = [trial.Trial(type = trial.TRIAL_TYPE_PL2, stimulus = './resources/narrative/narrative.wav', preload_stimulus = sound.Sound('./resources/narrative/narrative.wav'))]

    # load stimulus file
    stimuli = pd.read_csv('./resources/stimuli.csv')

    # find audio files for passive listening of words task
    vocoded = []
    pairs = []

    for i in np.arange(0, len(stimuli), 1):
        # grab item
        pair = stimuli.loc[i]
        file = pair.file.split('/')[-1]
        
        if pair.is_control:
            # handle control items (find p)
            p = file.split('_')[0]
            #file = f'./audio_vocoded/{file}'
            file = './audio_clear/' + ('_'.join(file.split('_')[0:2])) + '.wav'
            
            # add to list
            if file not in vocoded: vocoded.append(file); pairs.append(p)
        else:
            '''
            # handle target items (takes 1-2, repetition & pair)
            t1, t2 = file.split('_')[1].split('-')
            r = int(file.split('_')[-2][1:])
            p = file.split('_')[0]

            ## make vocoded file name
            #f1 = f'./audio_vocoded/{p}_T{t1}_12ch_r{r}_cs.wav'
            #f2 = f'./audio_vocoded/{p}_D{t2}_12ch_r{r}_cs.wav'
            f1 = f'./audio_clear/{p}_T{t1}.wav'
            f2 = f'./audio_clear/{p}_D{t2}.wav'

            # add to list
            if f1 not in vocoded: vocoded.append(f1); pairs.append(p)
            if f2 not in vocoded: vocoded.append(f2); pairs.append(p)
            '''

            # find items
            t1, t2 = file.split('_')[1].split('-')
            r = int(file.split('_')[-2][1:])
            p = file.split('_')[0]

            # make files, now including _all_ available versions
            ft1 = f'./audio_clear/{p}_T1.wav'
            ft2 = f'./audio_clear/{p}_T2.wav'
            fa1 = f'./audio_clear/{p}_D1.wav'
            fa2 = f'./audio_clear/{p}_D2.wav'

            for f_i in [ft1, ft2, fa1, fa2]:
                if f_i not in vocoded: vocoded.append(f_i); pairs.append(p)
    
    # setup randomisation for passive listening of word task
    randomised_vocoded = []
    randomised_pairs = []
    
    # keep track of consumed trials
    consumed = []

    # loop over trials
    i: int = 0
    while i < len(pairs):
        # grab current no and look-back
        j = i
        L_back = __FLAG_MAX_BUFFER if j >= __FLAG_MAX_BUFFER else j

        # setup search
        max_n = 100
        cur_n = 0
        valid = False

        # perform search
        while not valid:
            # increment
            cur_n += 1
            
            # reset (if necessary)
            if cur_n > max_n:
                randomised_vocoded = []
                randomised_pairs = []
                consumed = []
                i = -1
                break
            
            # draw an item
            opts = [opt for opt in np.arange(0, len(pairs), 1) if opt not in consumed]
            draw = np.random.choice(opts)
            draw_v = vocoded[draw]
            draw_p = pairs[draw]
            valid = True
            
            # control token repetition within buffer length
            for L in np.arange(j-1, j-L_back-1, -1):
                if (randomised_pairs[L] == draw_p):
                    valid = False
                    break
            
            # skip if not valid
            if not valid: continue
            
            # consume
            consumed.append(draw)
            randomised_vocoded.append(draw_v)
            randomised_pairs.append(draw_p)

        i += 1

    # make PL trials
    passive_listening_words = [trial.Trial(type = trial.TRIAL_TYPE_PL1, stimulus = file, preload_stimulus = sound.Sound(file)) for file in randomised_vocoded]
    
    # setup identities, features and contexts
    identities = ['face3', 'face6', 'face7', 'face10', 'face11', 'face12']
    features = ['glasses3', 'glasses2', 'piercing1', 'piercing2', 'scar1', 'scar2']
    contexts = ['essen', 'fashion', 'unterhaltung', 'technik', 'politik', 'outdoor']
    stories = {'essen': 'Ich bin der Chef eines Restaurants.\nEssen ist meine Leidenschaft und\nin der Küche fühle ich mich wohl.\n ',
               'fashion': 'Ich interessiere mich für Klamotten\nund Style. Heute leite ich\nein erfolgreiches Mode-Label.\n ',
               'unterhaltung': 'Ich bin als Kritiker tätig.\nDabei beschäftige ich mich mit Kunst\naller Art, hauptsächlich mit\nTheater, Film und Musik.', 
               'technik': 'Ich finde Computer und Technik\nspannend. Ich leite eine Firma\nfür Informationstechnologie\nund Elektronik.',
               'politik': 'Ich bin sehr interessiert an\nPolitik und verschiedenen Regierungs-\nformen. Ich arbeite im Parlament.\n ',
               'outdoor': 'Ich reise gern und bin am\nliebsten in der Natur.\nIch interessiere mich für Pflanzen,\nTiere und Sport.'}
    
    # remix identity/feature and speaker/context mapping
    identities = np.random.choice(identities, size = (len(identities),), replace = False)
    features = np.random.choice(features, size = (len(features),), replace = False)
    speakers = [f'{identity}_{feature}' for identity, feature in zip(identities, features)]
    cmapping = {context: speaker for context, speaker in zip(contexts, speakers)}

    # setup practice trials
    practice = [trial.Trial(context = 'essen', speaker = cmapping['essen'], preload_speaker = visual.ImageStim(win, image = f'./resources/images/{cmapping["essen"]}.png', pos = [0, -0.75], depth = 0), stimulus = './audio_morphed/Mehl-mal_1-1_12ch_0.50_r1_cs.wav', preload_stimulus = sound.Sound('./audio_morphed/Mehl-mal_1-1_12ch_0.50_r1_cs.wav'), type = trial.TRIAL_TYPE_PRACTICE, no = -1, block = -1, options = ('Mehl', 'mal'), target_position = np.random.choice(['down', 'up'])),
                trial.Trial(context = 'unterhaltung', speaker = cmapping['unterhaltung'], preload_speaker = visual.ImageStim(win, image = f'./resources/images/{cmapping["unterhaltung"]}.png', pos = [0, -0.75], depth = 0), stimulus = './audio_morphed/Goethe-Göre_2-1_12ch_0.50_r1_cs.wav', preload_stimulus = sound.Sound('./audio_morphed/Goethe-Göre_2-1_12ch_0.50_r1_cs.wav'), type = trial.TRIAL_TYPE_PRACTICE, no = -1, block = -1, options = ('Goethe', 'Göre'), target_position = np.random.choice(['down', 'up'])),
                trial.Trial(context = 'politik', speaker = cmapping['politik'], preload_speaker = visual.ImageStim(win, image = f'./resources/images/{cmapping["politik"]}.png', pos = [0, -0.75], depth = 0), stimulus = './audio_morphed/Bund-Mund_2-2_12ch_0.50_r1_cs.wav', preload_stimulus = sound.Sound('./audio_morphed/Bund-Mund_2-2_12ch_0.50_r1_cs.wav'), type = trial.TRIAL_TYPE_PRACTICE, no = -1, block = -1, options = ('Bund', 'Mund'), target_position = np.random.choice(['down', 'up'])),
                trial.Trial(context = 'technik', speaker = cmapping['technik'], preload_speaker = visual.ImageStim(win, image = f'./resources/images/{cmapping["technik"]}.png', pos = [0, -0.75], depth = 0), stimulus = './audio_morphed/Starkstrom-Symptom_2-2_12ch_0.50_r1_cs.wav', preload_stimulus = sound.Sound('./audio_morphed/Starkstrom-Symptom_2-2_12ch_0.50_r1_cs.wav'), type = trial.TRIAL_TYPE_PRACTICE, no = -1, block = -1, options = ('Starkstrom', 'Symptom'), target_position = np.random.choice(['down', 'up'])),
                trial.Trial(context = 'fashion', speaker = cmapping['fashion'], preload_speaker = visual.ImageStim(win, image = f'./resources/images/{cmapping["fashion"]}.png', pos = [0, -0.75], depth = 0), stimulus = './audio_morphed/modern-ungern_1-1_12ch_0.50_r1_cs.wav', preload_stimulus = sound.Sound('./audio_morphed/modern-ungern_1-1_12ch_0.50_r1_cs.wav'), type = trial.TRIAL_TYPE_PRACTICE, no = -1, block = -1, options = ('modern', 'ungern'), target_position = np.random.choice(['down', 'up'])),
                trial.Trial(context = 'outdoor', speaker = cmapping['outdoor'], preload_speaker = visual.ImageStim(win, image = f'./resources/images/{cmapping["outdoor"]}.png', pos = [0, -0.75], depth = 0), stimulus = './audio_morphed/walken-borgen_2-2_12ch_0.50_r1_cs.wav', preload_stimulus = sound.Sound('./audio_morphed/walken-borgen_2-2_12ch_0.50_r1_cs.wav'), type = trial.TRIAL_TYPE_PRACTICE, no = -1, block = -1, options = ('walken', 'borgen'), target_position = np.random.choice(['down', 'up']))]
    
    # remix practice trial order
    practice = np.random.choice(practice, size = (len(practice),), replace = False)

    # setup data structures
    trials = []
    last_t = None
    last_n = 0

    # keep track of consumed trials
    consumed = []

    # loop over trials
    i: int = 0
    while i < len(stimuli):
        # grab current no and look-back
        j = i
        L_back = __FLAG_MAX_BUFFER if j >= __FLAG_MAX_BUFFER else j

        # setup search
        max_n = 100
        cur_n = 0
        valid = False

        # perform search
        while not valid:
            # increment
            cur_n += 1
            
            # reset (if necessary)
            if cur_n > max_n:
                trials = []
                consumed = []
                last_t = None
                last_n = 0
                r = 0
                i = -1
                break
            
            # draw a trial
            opts = [opt for opt in np.arange(0, len(stimuli), 1) if opt not in consumed]
            draw = np.random.choice(opts)
            draw_S = stimuli.loc[draw]
            T = trial.Trial(context = draw_S.context,
                            speaker = cmapping[draw_S.context],
                            preload_speaker = visual.ImageStim(win, image = f'./resources/images/{cmapping[draw_S.context]}.png', pos = [0, -0.75], depth = 0),
                            stimulus = draw_S[f'file'],
                            preload_stimulus = sound.Sound(draw_S[f'file']),
                            type = trial.TRIAL_TYPE_MAIN if not draw_S.is_control else trial.TRIAL_TYPE_CONTROL,
                            no = j,
                            block = j // __FLAG_BLOCK_LENGTH + 1,
                            options = (draw_S.target, draw_S.popular),
                            target_position = np.random.choice(['down', 'up']))
            valid = True

            # control same type repetitions (min)
            if T.speaker != last_t and last_n < __FLAG_MIN_REPETITIONS and i > 0:
                valid = False
                continue
            
            # control same type repetitions (max)
            if T.speaker == last_t and last_n >= __FLAG_MAX_REPETITIONS:
                valid = False
                continue
            
            # control token repetition within buffer length
            for L in np.arange(j-1, j-L_back-1, -1):
                if (T.options[0] == trials[L].options[0]) or (T.options[0] == trials[L].options[1]) or \
                   (T.options[1] == trials[L].options[0]) or (T.options[1] == trials[L].options[1]):
                    valid = False
                    break
            
            # skip if not valid
            if not valid: continue

            # update states
            if last_t != T.speaker:
                last_t = T.speaker
                last_n = 0
            
            # consume
            last_n += 1
            consumed.append(draw)
            trials.append(T)

        i += 1
    
    '''
    Start a new session
    '''

    # generate PID
    pid = ''
    while len(pid) == 0:
        pid = ''.join(np.random.choice(list('abcdefhijklmnopqrstuvwxyz0123456789'), size = (__FLAG_PID_LENGTH,))).upper()
        if os.path.isfile(os.path.join(__FLAG_DATA, pid + '.csv')): pid = ''
    
    # start data stream
    data = data.Dataset(pid, __FLAG_DATA)

    # start port
    port = trigger.Port(win, name = __FLAG_TRIGGER_PORT, simulate = __FLAG_TRIGGER_SIMULATE)

    # setup RTFE model
    rtfe = model.RTFE(pid)

    '''
    Commence experiment for participant
    '''

    # passive listening of individual words
    if __FLAG_SKIP_AHEAD < __FLAG_SKIP_AHEAD_PL1:
        # passive listening, instructions for second part
        if not __FLAG_FAST_MODE:
            trial.instruction(['Teil 1/5: Einweisung\n\n' + 
                                'Willkommen zur ersten Aufgabe des Experiments. ' + 
                                    'Sie werden im Folgenden Teil in jedem Durchgang zunächst ein Fixationskreuz sehen. ' + 
                                    'Daraufhin werden Sie ein Wort hören. ' + 
                                    'Ihre Aufgabe besteht zunächst darin, sich auf das gehörte Wort zu konzentrieren.\n\n' + 
                                'Achtung: Manchmal werden Ihnen nach dem Wort zwei Optionen angezeigt. ' +
                                    'Wählen Sie dann bitte die Option aus, die Sie gehört haben. ' + 
                                    'Drücken Sie bitte ^ (Pfeiltaste oben) für die obere und v (Pfeiltaste unten) für die untere Option.\n\n' +
                                'Bitte beachten Sie dabei, dass es wichtig ist, dass Sie während jedem Durchgang so ruhig wie möglich sitzen. ' + 
                                    'Bewegen Sie darüberhinaus Ihre Augen während eines Durchgangs bitte nicht. ' + 
                                    'Fokussieren Sie stattdessen das Fixationskreuz und blinzeln in den kurzen Pausen zwischen den Durchgängen.\n\n' + 
                                'Sie werden während dieser Aufgabe in regelmäßigen Abständen die Möglichkeit einer Pause angeboten bekommen. ' + 
                                    'Die ganze Aufgabe wird ungefähr 7 Minuten dauern.\n\n' + 
                                'Sind Sie bereit?\n\n' + 
                                'Drücken Sie -> (Pfeiltaste rechts), um mit der Aufgabe zu beginnen.'], win)
            trial.countdown(win, L = __FLAG_COUNTDOWN_L)
        
        # passive listening task
        for i, T in enumerate(passive_listening_words):
            if __FLAG_VERBOSE: print(f'PL1: {i+1}/{len(passive_listening_words)} - {global_clock.getTime()}')
            if i % __FLAG_BLOCK_LENGTH == 0 and i > 0: 
                if __FLAG_VERBOSE: print(f'PL1: BREAK')
                trial.instruction([f'{i}/{len(passive_listening_words)}: Pause\n\n' + 
                                   'Wenn Sie möchten, können Sie nun eine kurze Pause nehmen.\n\n' + 
                                   'Drücken Sie -> (Pfeiltaste rechts), um fortzufahren.'], win)
                trial.countdown(win, L = __FLAG_COUNTDOWN_L)

            # should this trial include an attention check?
            do_check = np.random.random() > (1 - __FLAG_PL_CHOICE_PROB)
            
            # start trial
            T.ts = global_clock.getTime()
            res = trial.natural_listening(T, win, port, persist = do_check)
            
            # for some trials, we want an attention check (foil task)
            if do_check:
                # make sure we still have some buffer time until 2afc starts (for decoding)
                core.wait(0.4)

                # find current stimulus, target and alternative & grid
                stim = T.stimulus.split('/')[-1]
                A, B = stim.split('_')[0].split('-')
                type_stim = stim.split('_')[1]
                target, alternative = (A, B) if type_stim[0] == 'T' else (B, A)
                t_pos = np.random.choice(['up', 'down'])
                up, down = (target, alternative) if t_pos == 'up' else (alternative, target)
                
                # draw box for contrast & fixation
                box = visual.Rect(win, 8, 2.5, 'deg', pos = [0, 0], fillColor = [1, 1, 1], fillColorSpace = 'rgb', opacity = 0.45, depth = 1)
                cross = visual.TextStim(win, pos = [0, 0], text = '+', color = [-1, -1, -1], colorSpace = 'rgb', depth = 2)
                
                # draw opts
                opt_down = visual.TextStim(win, pos = [0, -0.75], text = down, color = [-0.5, -0.5, -0.5], depth = 2, units = 'deg', height = 0.85)
                opt_up = visual.TextStim(win, pos = [0, 0.75], text = up, color = [-0.5, -0.5, -0.5], depth = 2, units = 'deg', height = 0.85)

                # draw
                box.draw()
                cross.draw()
                opt_down.draw()
                opt_up.draw()

                # flip and send trigger
                win.callOnFlip(port.send, trial.TRIGGER_TRIAL_RESPONSE)
                win.flip()

                # get response
                choice, rt = trial.get_response(valid = ['down', 'up'], allow_exit = True)

                # add to trial data
                res.correct = t_pos == choice
                res.rt = rt

                # flip and send trigger
                win.callOnFlip(port.send, trial.TRIGGER_TRIAL_RESPONSE_POST)
                win.flip()
                core.wait(0.25)

            data.write(res)


    # passive listening task
    if __FLAG_SKIP_AHEAD < __FLAG_SKIP_AHEAD_PL2:
        # passive listening instructions
        if not __FLAG_FAST_MODE:
            trial.instruction(['Teil 2/5: Einweisung\n\n' + 
                                'Sie haben die erste Aufgabe abgeschlossen. Super!\n\n' + 
                                'Die folgende Aufgabe wird sehr ähnlich sein. ' + 
                                    'Dieses mal werden Sie allerdings ' + ('eine Geschichte' if len(passive_listening) == 1 else 'zwei Geschichten') + ' hören. ' + 
                                    'Ihre Aufgabe besteht erneut darin, gut zuzuhören. ' + 
                                    'Beachten Sie dabei, dass Sie dieses Mal nach dem Hören einige inhaltliche Fragen beantworten werden müssen.\n\n' + 
                                'Bitte beachten Sie auch, dass es wichtig ist, dass Sie so ruhig wie möglich sitzen und Ihre Augen so wenig wie möglich bewegen sollten. ' + 
                                    'Fokussieren Sie erneut das Fixationskreuz in der Mitte des Bildschirms.\n\n' + 
                                'Sie werden während dieser Aufgabe keine Pause nehmen können. ' + 
                                    'Die ganze Aufgabe wird ungefähr 5 Minuten dauern.\n\n' + 
                                'Sind Sie bereit?\n\n' + 
                                'Drücken Sie -> (Pfeiltaste rechts), um mit der Aufgabe zu beginnen.'], win)
            trial.countdown(win, L = __FLAG_COUNTDOWN_L)
        
        # passive listening task
        for i, T in enumerate(passive_listening):
            if __FLAG_VERBOSE: print(f'PL2: {i+1}/{len(passive_listening)} - {global_clock.getTime()}')
            if i > 0:
                if __FLAG_VERBOSE: print(f'PL2: BREAK')
                trial.instruction(['Teil 2/5: Pause\n\n' + 
                                    'Wenn Sie möchten, können Sie nun eine kurze Pause nehmen bevor es mit der nächsten Geschichte weitergeht.\n\n' + 
                                    'Drücken Sie -> (Pfeiltaste rechts), um fortzufahren.'], win)
                trial.countdown(win, L = __FLAG_COUNTDOWN_L)
            
            # play story
            T.ts = global_clock.getTime()
            res = trial.natural_listening(T, win, port)
            data.write(res)

            # ask question(s)
            T = trial.survey(trial.Trial(no = -10*i+0, type = trial.TRIAL_TYPE_SURVEY, stimulus = '1/3: Womit waren die Wände des Brunnens bedeckt?'),
                         [(visual.TextStim(win, pos = [0, -5.75], text = 'Küchenschränke und Bücher (Pfeiltaste runter)', color = [-1, -1, -1], colorSpace = 'rgb', units = 'deg', height = 0.85), 'down'),
                          (visual.TextStim(win, pos = [0, -4.25], text = 'Dreck und alte Farbe (Pfeiltaste hoch)', color = [-1, -1, -1], colorSpace = 'rgb', units = 'deg', height = 0.85), 'up')], win)
            data.write(T)

            T = trial.survey(trial.Trial(no = -10*i+1, type = trial.TRIAL_TYPE_SURVEY, stimulus = '2/3: Wie weit glaubte Alice gefallen zu sein?'),
                         [(visual.TextStim(win, pos = [0, -5.75], text = '1250 Meilen (Pfeiltaste runter)', color = [-1, -1, -1], colorSpace = 'rgb', units = 'deg', height = 0.85), 'down'),
                          (visual.TextStim(win, pos = [0, -4.25], text = '850 Meilen (Pfeiltaste hoch)', color = [-1, -1, -1], colorSpace = 'rgb', units = 'deg', height = 0.85), 'up')], win)
            data.write(T)

            T = trial.survey(trial.Trial(no = -10*i+2, type = trial.TRIAL_TYPE_SURVEY, stimulus = '3/3: Wem folgte Alice in den Brunnen?'),
                         [(visual.TextStim(win, pos = [0, -5.75], text = 'Ihrer Schwester (Pfeiltaste runter)', color = [-1, -1, -1], colorSpace = 'rgb', units = 'deg', height = 0.85), 'down'),
                          (visual.TextStim(win, pos = [0, -4.25], text = 'Einem Kaninchen (Pfeiltaste hoch)', color = [-1, -1, -1], colorSpace = 'rgb', units = 'deg', height = 0.85), 'up')], win)
            data.write(T)



    # instructions & practice part
    if __FLAG_SKIP_AHEAD < __FLAG_SKIP_AHEAD_PRACTICE:
        # main instructions part1
        if not __FLAG_FAST_MODE: 
            trial.instruction(['Teil 3/5: Einweisung\n\n' + 
                                    'Sie haben die ersten beiden Aufgabe abgeschlossen. Klasse!\n\n' + 
                                    f'Stellen Sie sich nun vor, Sie gehen auf eine Feier, auf der Sie {len(contexts)} Personen kennenlernen. ' + 
                                        'In jedem Durchgang sehen Sie eine dieser Personen. ' + 
                                        'Die Person sagt ein Wort, das aufgrund der Umgebung kaum verständlich ist. ' +
                                        'Ihre Aufgabe ist es nun, aus zwei gezeigten Optionen das Wort auszuwählen, das die Person am ehesten gesagt hat. ' + 
                                        'Dabei wird jede Person genau ein Thema haben, über welches sie spricht.\n\n' + 
                                    f'Auf der folgenden Seite sehen Sie alle {len(contexts)} Personen, die Sie auf der Feier kennenlernen werden. ' + 
                                        'Versuchen Sie kurz, sich mit diesen ein wenig vertraut zu machen. ' + 
                                        'Sobald Sie fertig sind, drücken Sie -> (Pfeiltaste rechts), umfortzufahren.\n\n' + 
                                    'Dieser Teil des Experiments wird in etwa 30 Minuten dauern.\n\n' + 
                                    'Drücken Sie -> (Pfeiltaste rechts), um zur nächsten Seite zu gelangen.'], win)

            images = []
            txts = []
            # speakers
            for i, mapping in enumerate(cmapping):
                spkr = cmapping[mapping]
                stry = stories[mapping]
                j = i // 3
                i = i % 3

                pos = [-0.5 + i*(1.5/3), -0.3 + j*0.85]

                img = visual.ImageStim(win, image = f'./resources/images/{spkr}.png', pos = pos, size = (0.2, 0.35), units = 'norm')
                img.draw()

                txt = visual.TextStim(win, pos = [pos[0], pos[1] - 0.3], text = stry, depth = 2, units = 'norm', height = 0.05)
                txt.draw()

                images.append(img)
                txts.append(txt)

            win.flip()
            trial.get_response(['right'], max_rt = None)

            # main instructions part2
            trial.instruction(['Teil 3/5: Einweisung\n\n' + 
                                    'Sobald die Optionen angezeigt werden, drücken Sie entweder v (Pfeiltaste unten) für die untere oder ^ (Pfeiltaste oben) für die obere Option. ' + 
                                        'Wenn Ihre Antwortmöglichkeiten beispielsweise `Bummel` (oben) und `Hummel` (unten) sind und Sie es für wahrscheinlicher halten, dass die Person `Hummel` gesagt hat, drücken Sie v.\n\n' + 
                                    'Für jede Antwort haben Sie zwei Sekunden Zeit. ' + 
                                        'Entscheiden Sie sich daher bitte so schnell und akkurat wie möglich. ' + 
                                        'Nach Ihrer Entscheidung erhalten Sie Feedback. ' + 
                                        'Ein Fehler wird rot und eine korrekte Entscheidung grün angezeigt. ' + 
                                        'Die Antwort wird automatisch übersprungen, sobald die Zeit abgelaufen ist.\n\n' + 
                                    'Achtung! Es ist wichtig, dass Sie während jedem Durchlauf Ihre Augen nicht bewegen und nicht blinzeln. ' + 
                                        'Sie werden während jedem Durchgang ein Fixationskreuz in der Mitte des Bildschirms sehen. ' + 
                                        'Fokussieren Sie sich bitte stets auf das Kreuz. ' + 
                                        'Die Personen und Antwortmöglichkeiten werden dennoch erkennbar sein.\n\n' + 
                                    'Sind Sie bereit? Wir beginnen mit sechs Übungsdurchläufen.\n\n' + 
                                    'Drücken Sie -> (Pfeiltaste rechts), um mit dem Übungsblock zu beginnen.'], win)
            trial.countdown(win, L = __FLAG_COUNTDOWN_L)

        # run practice trials
        for i, T in enumerate(practice):
            if __FLAG_VERBOSE: print(f'PT: {i+1}/{len(practice)} - {global_clock.getTime()}')
            T.ts = global_clock.getTime()
            res = trial.tafc_audio_visual(T, win, port)
            choice = res.options[0].lower() if res.correct else res.options[1].lower()
            rtfe.step(-(len(practice)-i), choice, res.context)
            data.write(res)
    
    # counter vars
    all_blocks = 0
    all_blocks_n = 0
    all_blocks_m = 0

    # instructions & main part
    if __FLAG_SKIP_AHEAD < __FLAG_SKIP_AHEAD_MAIN:
        # main instructions part3
        if not __FLAG_FAST_MODE: 
            trial.instruction(['Teil 3/5: Einweisung\n\n' + 
                                'Sie haben den Übungsblock abgeschlossen. Vielleicht sind Ihnen manche Entscheidungen dabein nicht leicht gefallen. ' + 
                                    'Zur Erinnerung: Jede Person hat genau ein Thema, über das sie spricht. ' + 
                                    'Nutzen Sie dieses Wissen, um das Wort, das die Person tatsächlich gesagt hat, zu identifizieren. ' + 
                                    'Bitte antworten Sie dabei so schnell und akkurat wie möglich.\n\n' + 
                                'Denken Sie bitte gleichermaßen daran, dass Sie Ihre Augen während eines Durchgangs nicht bewegen sollen. ' + 
                                    'Fokussieren Sie stattdessen das Fixationskreuz in der Mitte des Bildschirms.\n\n' + 
                                'Sind Sie bereit?\n\n' + 
                                'Drücken Sie -> (Pfeiltaste rechts), um mit der Aufgabe zu beginnen.'], win)
            trial.countdown(win, L = __FLAG_COUNTDOWN_L)

        last_block = 0
        # run main trials
        for i, T in enumerate(trials):
            if __FLAG_VERBOSE:
                total = 0.0 if all_blocks_n < 1 else np.round(all_blocks / all_blocks_n * 100, 2)
                last = np.round(last_block / __FLAG_BLOCK_LENGTH * 100, 2)
                print(f'MT: {i+1}/{len(trials)} - {int(global_clock.getTime())}s - Block: {last}% - Total: {total}% - Misses: {all_blocks_m}')
            
            # add break (if necessary)
            if i % __FLAG_BLOCK_LENGTH == 0 and i > 0:
                if __FLAG_VERBOSE: print(f'MT: BREAK')
                last_block = np.round(last_block / __FLAG_BLOCK_LENGTH * 100, 2)
                trial.instruction([f'{i}/{len(trials)} Pause\n\n' + 
                                        f'Im letzten Block haben Sie in {last_block}% der Fälle das von der Person geäusserte Wort korrekt erkannt. ' + 
                                            'Achtung: Jede Person hat genau ein Thema, über welches sie spricht.\n\n' + 
                                        'Wenn Sie bereit sind, dann drücken Sie -> (Pfeiltaste rechts), um fortzufahren.'], win)
                trial.countdown(win, L = __FLAG_COUNTDOWN_L)
                last_block = 0

            # run trial
            T.ts = global_clock.getTime()
            res = trial.tafc_audio_visual(T, win, port)
            choice = res.options[0].lower() if res.correct else res.options[1].lower()
            rtfe.step(i, choice, res.context)
            data.write(res)
            last_block += float(res.correct)
            all_blocks += float(res.correct)
            all_blocks_n += 1
            if res.choice not in ['down', 'up']: all_blocks_m += 1
    
    '''
    Real-time free-energy model evaluation for
    manipulation of final task

    NOTE: This is a bit more involved and so,
    to make this more legible: Basically, we
    are trying to find out how well all pairs
    would match to either available context.
    From this, we sample 120 new items (10 
    excellent, 10 terrible exemplars per
    context_A but 10 terrible and 10 excellent
    in context_B, respectively) and preload
    the new stimuli.

    NOTE: This may sound simple, but unfortunately
    this is actually an undecideable problem. Be aware
    that there is some wacky code and hackery ahead.
    '''

    # setup loading screen
    trial.loading(win)
    
    # find all potential targets (and associated files) for oddballs
    items = []
    altes = []
    files = []
    for i in np.arange(0, len(stimuli), 1):
        # grab item
        pair = stimuli.loc[i]
        file = pair.file.split('/')[-1]
        
        if pair.is_control:
            # handle control items (find p)
            t1, t2 = file.split('_')[0].split('-')
            file = f'./audio_vocoded/{file}'
            
            # add to list
            if t1 not in items: items.append(t1); altes.append(t2); files.append(file)
        else:
            # handle target items (takes 1-2, repetition & pair)
            t1, t2 = file.split('_')[1].split('-')
            r = int(file.split('_')[-2][1:])
            p = file.split('_')[0]
            p1, p2 = p.split('-')

            # make vocoded file name
            f1 = f'./audio_vocoded/{p}_T{t1}_12ch_r{r}_cs.wav'
            f2 = f'./audio_vocoded/{p}_D{t2}_12ch_r{r}_cs.wav'

            # add to list
            if p1 not in items: items.append(p1); altes.append(p2); files.append(f1)
            if p2 not in items: items.append(p2); altes.append(p1); files.append(f2)
    
    # evaluate how well each item would fit semantic space of participant
    df_item = []
    df_alte = []
    df_con1 = []
    df_fit1 = []
    df_con2 = []
    df_fit2 = []
    df_delt = []
    df_file = []
    for item, alte, file in zip(items, altes, files):
        # loop over context (as main)
        for c1 in contexts:
            # retrieve prior
            p1 = rtfe.p_mu_s[:,rtfe.speakers == c1].squeeze()

            # loop over context (as alt)
            for c2 in contexts:
                # skip if same contexts
                if c1 == c2: continue

                # retrieve prior
                p2 = rtfe.p_mu_s[:,rtfe.speakers == c2].squeeze()

                # compute semantic fit
                m1 = np.dot(rtfe.G[item.lower()], p1) / (np.linalg.norm(rtfe.G[item.lower()]) * np.linalg.norm(p1))
                m2 = np.dot(rtfe.G[item.lower()], p2) / (np.linalg.norm(rtfe.G[item.lower()]) * np.linalg.norm(p2))

                # add data
                df_item.append(item)
                df_alte.append(alte)
                df_con1.append(c1)
                df_fit1.append(m1)
                df_con2.append(c2)
                df_fit2.append(m2)
                df_delt.append(m1 - m2)
                df_file.append(file)
    
    # rank items by delta (descending)
    df_rank = np.array(df_delt).argsort()[::-1]
    preselection = np.array(df_delt)[df_rank] >= 0

    # create data frame from ranked preselection
    df = pd.DataFrame.from_dict({'word': np.array(df_item)[df_rank][preselection], 'alternative': np.array(df_alte)[df_rank][preselection], 'context1': np.array(df_con1)[df_rank][preselection], 
                                 'fit1': np.array(df_fit1)[df_rank][preselection], 'context2': np.array(df_con2)[df_rank][preselection], 'fit2': np.array(df_fit2)[df_rank][preselection], 
                                 'delta': np.array(df_delt)[df_rank][preselection], 'file': np.array(df_file)[df_rank][preselection]})
    
    # start sampling good and bad exemplars
    # NOTE: We are doing this in a stochastic
    # manner (albeit mildly) to avoid a scenario
    # where no real solution (that satisfies that
    # all targets be crossed in contexts and all
    # contexts be of full length) can be found.
    #
    # NOTE: This is some seriously horrible code,
    # but it works. Replaced the earlier version
    # of sampling that performed well _most_ of
    # the time, but would sometimes have outliers
    # that took 5-10mins to solve with something
    # that, in the absolute worst case, should
    # take no longer than 10s (typically 500ms).
    # This new version is extensively tested on
    # the data sets we have from previous exps.
    # I don't love this, but I seriously tried
    # approaching this with optim theory and
    # I just couldn't find a good solution, so
    # sampling it is. At the very least, this 
    # _will_ converge timely and create good
    # results.
    done = False
    oddballs = []
    oddballs_unordered = []
    counts_good = {context: 0 for context in contexts}
    counts_bad = {context: 0 for context in contexts}
    consumed_good = []; consumed_bad = []
    cdf = {context: df.loc[df.context1 == context].reset_index(drop = True) for context in contexts}
    builds = 0

    while not done:
        builds += 1
        if __FLAG_VERBOSE: print(f'Loading OT: build{builds} - {global_clock.getTime()}')

        # make sure we reset appropriately
        if not done:
            oddballs_unordered = []; oddballs = []; consumed_good = []; consumed_bad = []
            counts_good = {context: 0 for context in contexts}
            counts_bad = {context: 0 for context in contexts}
            cdf = {context: df.loc[df.context1 == context].reset_index(drop = True) for context in contexts}
        
        full_run = False
        last_con = contexts[0]
        while not full_run:
            # check if we can exit
            full_run = True
            for context in contexts:
                if counts_good[context] < __FLAG_ODDBALL_CATEGORY: full_run = False; break
                if counts_bad[context] < __FLAG_ODDBALL_CATEGORY: full_run = False; break
            if len(cdf[last_con]) < 1: full_run = True; break
            if full_run: continue

            # grab item
            draw = cdf[last_con].iloc[0]

            # check whether it is still required
            if (counts_good[draw.context1] >= __FLAG_ODDBALL_CATEGORY) or \
               (counts_bad[draw.context2] >= __FLAG_ODDBALL_CATEGORY):
                cdf[last_con] = cdf[last_con].drop([0]).reset_index(drop = True)
                continue
            
            # check if item has been used before
            if (draw.word in consumed_good) or (draw.word in consumed_bad) or \
               (draw.alternative in consumed_good) or (draw.alternative in consumed_bad):
                cdf[last_con] = cdf[last_con].drop([0]).reset_index(drop = True)
                continue
            
            # make sure item is not dubious fit (should not be possible anyway but let's make sure)
            if draw.fit1 < draw.fit2:
                cdf[last_con] = cdf[last_con].drop([0]).reset_index(drop = True)
                continue
            
            # add stochasticity to sampler
            skip = np.random.choice([0, 1], p = [0.5, 0.5]).astype(bool)
            if skip:
                cdf[last_con] = cdf[last_con].drop([0]).reset_index(drop = True)
                continue
            
            # consume item
            consumed_good.append(draw.word)
            consumed_bad.append(draw.word)
            counts_good[draw.context1] += 1
            counts_bad[draw.context2] += 1
            oddballs_unordered.append(draw)

            # propagate consumption
            for con in cdf:
                cdf[con] = cdf[con].drop(np.where((cdf[con].word == draw.word) | (cdf[con].word == draw.alternative) | (cdf[con].alternative == draw.word) | (cdf[con].alternative == draw.alternative))[0]).reset_index(drop = True)

            # switch context
            last_con = min(counts_good, key = counts_good.get)
        
        # determine if we were successful, otherwise resample
        done = True
        for context in contexts:
            if counts_good[context] != __FLAG_ODDBALL_CATEGORY or counts_bad[context] != __FLAG_ODDBALL_CATEGORY:
                done = False
                break
        if not done: continue
        
        # pseudorandomise order of presentation
        last_t, last_c = None, None
        last_n, last_k = 0, 0
        consumed_good = []; consumed_bad = []
        i = 0
        while i < (6*__FLAG_ODDBALL_LENGTH):
            # setup look back
            j = i
            L_back = __FLAG_MAX_BUFFER if j >= __FLAG_MAX_BUFFER else j

            # setup search
            max_n = 100
            cur_n = 0
            valid = False

            # perform search
            while not valid:
                # increment
                cur_n += 1
                
                # reset (if necessary)
                if cur_n > max_n:
                    oddballs = []
                    consumed_good = []; consumed_bad = []
                    last_t, last_c = None, None
                    last_n, last_k = 0, 0
                    i = -1
                    break
                
                # flip coin for good/bad
                coin = np.random.choice([0, 1])

                # skip if too many condition repetitions
                if last_c == coin and last_k >= __FLAG_MAX_REPETITIONS: continue

                # generate options and draw
                opts = [opt for opt in np.arange(0, len(oddballs_unordered), 1) if ((oddballs_unordered[opt].word not in consumed_good) or (oddballs_unordered[opt].word not in consumed_bad))]
                draw = np.random.choice(opts)
                draw = oddballs_unordered[draw]
                valid = True

                # skip if this condition was already met
                if (coin == 1) and (draw.word in consumed_good): valid = False; continue
                elif (coin == 0) and (draw.word in consumed_bad): valid = False; continue

                # set new context
                context = draw.context1 if coin == 1 else draw.context2

                # skip if too many context repetitions
                if (context == last_t) and (last_n >= __FLAG_MAX_REPETITIONS): valid = False; continue

                # control token repetitions within buffer length
                for L in np.arange(j-1, j-L_back-1, -1):
                    if (draw.word == oddballs[L].options[0]) or (draw.word == oddballs[L].options[1]) or \
                       (draw.alternative == oddballs[L].options[0]) or (draw.alternative == oddballs[L].options[1]):
                        valid = False
                        break
                
                # skip if invalid
                if not valid: continue

                # update condition states
                if last_c != coin:
                    last_c = coin
                    last_k = 0
                last_k += 1

                # update context states
                if last_t != context:
                    last_t = context
                    last_n = 0
                last_n += 1

                # consume
                if coin == 1: consumed_good.append(draw.word)
                else: consumed_bad.append(draw.word)
                oddballs.append(trial.Trial(context = context,
                                            speaker = cmapping[draw.context1],
                                            preload_speaker = visual.ImageStim(win, image = f'./resources/images/{cmapping[context]}.png', pos = [0, -0.75], depth = 0),
                                            stimulus = draw.file,
                                            preload_stimulus = sound.Sound(draw.file),
                                            type = trial.TRIAL_TYPE_ODD_GOOD if coin == 1 else trial.TRIAL_TYPE_ODD_BAD,
                                            no = j,
                                            block = j // __FLAG_BLOCK_LENGTH + 1,
                                            options = (draw.word, draw.alternative),
                                            target_position = np.random.choice(['down', 'up'])))
            i += 1
    
    # log data frame (for QC later)
    df.to_excel(f'./rtfe/{pid}.xlsx')

    '''
    Commence final task for participant
    '''

    # instructions and oddball
    if __FLAG_SKIP_AHEAD < __FLAG_SKIP_AHEAD_ODDBALL:
        # instructions for oddball
        if not __FLAG_FAST_MODE:
            total = np.round(all_blocks / float(all_blocks_n) * 100, 2)
            trial.instruction(['Teil 4/5: Einweisung\n\n' + 
                                f'Super! Sie haben die Aufgabe mit insgesamt {total}% abgeschlossen.\n\n' + 
                                'Im folgenden Teil werden Sie erneut die Personen sehen und Worte sprechen hören. ' + 
                                    'Auch hier wird Ihre Aufgabe sein, das Wort auszuwählen, das die Person am Ehesten gesagt hat. ' + 
                                    'Dieses mal werden Sie allerdings kein Feedback nach dem jeweiligen Durchlauf erhalten. ' + 
                                    'Denken Sie daran, dass jede Person über genau ein Thema spricht.\n\n' + 
                                'Denken Sie auch daran, dass Sie Ihre Augen während jedem Durchgang nicht bewegen sollen. ' + 
                                    'Fokussieren Sie stattdessen das Fixationskreuz.\n\n' + 
                                'Bitte bedenken Sie ebenfalls, dass die Antworten so schnell und akkurat wie möglich gegeben werden sollen.\n\n' + 
                                'Dieser Teil wird ungefähr 15 Minuten dauern.\n\n' + 
                                'Drücken Sie -> (Pfeiltaste rechts), um mit der Aufgabe zu beginnen.'], win)
            trial.countdown(win, L = __FLAG_COUNTDOWN_L)
        
        # run oddball trials
        for i, T in enumerate(oddballs):
            if __FLAG_VERBOSE: print(f'OT: {i+1}/{len(oddballs)} - {global_clock.getTime()}')
            # add break (if necessary)
            if i % __FLAG_BLOCK_LENGTH == 0 and i > 0: 
                if __FLAG_VERBOSE: print(f'OT: BREAK')
                trial.instruction([f'{i}/{len(oddballs)}: Pause\n\n' + 
                                    'Wenn Sie möchten, nehmen Sie nun eine kurze Pause.\n\n' + 
                                    'Sobald Sie mit der Aufgabe weitermachen möchten, drücken Sie auf -> (Pfeiltaste rechts).'], win)
                trial.countdown(win, L = __FLAG_COUNTDOWN_L)

            # run trial
            T.ts = global_clock.getTime()
            res = trial.tafc_audio_visual(T, win, port, feedback = False)
            choice = res.options[0].lower() if res.correct else res.options[1].lower()
            rtfe.step(len(trials) + i, choice, res.context)
            data.write(res)
    
    # post-experiment survey
    if __FLAG_SKIP_AHEAD < __FLAG_SKIP_AHEAD_SURVEY:
        if not __FLAG_FAST_MODE:
            trial.instruction(['Teil 5/5: Einweisung\n\n' + 
                                'Sie haben nun fast alle Aufgaben abgeschlossen. Super!\n\n' + 
                                'Es folgen nun noch einige Fragen zum Verlauf des Experiments. ' + 
                                    'Diese können Sie wieder mit v (Pfeiltaste unten) und ^ (Pfeiltaste oben) beantworten.\n\n' + 
                                'Sind Sie bereit?\n\n' + 
                                'Drücken Sie -> (Pfeiltaste rechts), um mit den Fragen zu beginnen.'], win)
        
        # Q1
        T = trial.survey(trial.Trial(no = 0, type = trial.TRIAL_TYPE_SURVEY, stimulus = 'Hatten Sie den Eindruck, dass bestimmte Personen besonders gerne über bestimmte Themengebiete gesprochen haben?'),
                         [(visual.TextStim(win, pos = [0, -5.75], text = 'Nein (Pfeiltaste runter)', color = [-1, -1, -1], colorSpace = 'rgb', units = 'deg', height = 0.85), 'down'),
                          (visual.TextStim(win, pos = [0, -4.25], text = 'Ja (Pfeiltaste hoch)', color = [-1, -1, -1], colorSpace = 'rgb', units = 'deg', height = 0.85), 'up')], win)
        data.write(T)

        # Q2
        T = trial.survey(trial.Trial(no = 1, type = trial.TRIAL_TYPE_SURVEY, stimulus = 'Hatten Sie den Eindruck, dass Sie sich manchmal gegen das gehörte Wort entscheiden mussten, um die richtige Antwort zu geben?'),
                         [(visual.TextStim(win, pos = [0, -5.75], text = 'Nein (Pfeiltaste runter)', color = [-1, -1, -1], colorSpace = 'rgb', units = 'deg', height = 0.85), 'down'),
                          (visual.TextStim(win, pos = [0, -4.25], text = 'Ja (Pfeiltaste hoch)', color = [-1, -1, -1], colorSpace = 'rgb', units = 'deg', height = 0.85), 'up')], win)
        data.write(T)

        # conditional Q3
        T3 = trial.Trial(no = 2, type = trial.TRIAL_TYPE_SURVEY, stimulus = 'Haben Sie in diesen Fällen eher angegeben, was Sie gehört zu haben glaubten oder was Sie für die richtige Antwort hielten?')
        if T.choice == 'up':
            T3 = trial.survey(T3,
                         [(visual.TextStim(win, pos = [0, -5.75], text = 'Eher das Gehörte (Pfeiltaste runter)', color = [-1, -1, -1], colorSpace = 'rgb', units = 'deg', height = 0.85, wrapWidth = 30), 'down'),
                          (visual.TextStim(win, pos = [0, -4.25], text = 'Eher die richtige Antwort (Pfeiltaste hoch)', color = [-1, -1, -1], colorSpace = 'rgb', units = 'deg', height = 0.85, wrapWidth = 30), 'up')], win)
            data.write(T3)

        # conditional Q4
        T4 = trial.Trial(no = 3, type = trial.TRIAL_TYPE_SURVEY, stimulus = 'Haben Sie diese Strategie im Laufe des Experiments bewusst gewechselt?')
        if T.choice == 'up':
            T4 = trial.survey(T4,
                         [(visual.TextStim(win, pos = [0, -5.75], text = 'Nein (Pfeiltaste runter)', color = [-1, -1, -1], colorSpace = 'rgb', units = 'deg', height = 0.85), 'down'),
                          (visual.TextStim(win, pos = [0, -4.25], text = 'Ja (Pfeiltaste hoch)', color = [-1, -1, -1], colorSpace = 'rgb', units = 'deg', height = 0.85), 'up')], win)
            data.write(T4)

        # conditional Q5
        T5 = trial.Trial(no = 4, type = trial.TRIAL_TYPE_SURVEY, stimulus = 'Welche der beiden Optionen beschreibt diesen Wechsel am Besten?')
        if T4.choice == 'up':
            T5 = trial.survey(T5,
                         [(visual.TextStim(win, pos = [0, -5.75], text = 'Zunächst `Eher das Gehörte`, dann `Eher die richtige Antwort`. (Pfeiltaste runter)', color = [-1, -1, -1], colorSpace = 'rgb', units = 'deg', height = 0.85, wrapWidth = 35), 'down'),
                          (visual.TextStim(win, pos = [0, -4.25], text = 'Zunächst `Eher die richtige Antwort`, dann `Eher das Gehörte`. (Pfeiltaste hoch)', color = [-1, -1, -1], colorSpace = 'rgb', units = 'deg', height = 0.85, wrapWidth = 35), 'up')], win)
            data.write(T5)

    # debrief participant
    if not __FLAG_FAST_MODE:
        trial.instruction(['Abschluss des Experiments\n\n' + 
                            'Sie haben das Experiment erfolgreich abgeschlossen. ' + 
                                'Wir bedanken uns herzlich für Ihre Teilnahme und Ihren Beitrag zur Forschung. ' + 
                                'Sollten Sie noch Fragen bezüglich des Experiments haben, wenden Sie sich bitte an die Experimentleitung.\n\n' + 
                            'Drücken Sie -> (Pfeiltaste rechts), um das Experiment zu beenden.'], win)