// String.format prototype
// as per fearphage on https://stackoverflow.com/a/4673436 (last accessed 28/12/2022 11:59)
if (!String.prototype.format) {
    String.prototype.format = function() {
      var args = arguments;
      return this.replace(/{(\d+)}/g, function(match, number) { 
        return typeof args[number] != 'undefined'
          ? args[number]
          : match
        ;
      });
    };
}

// 2afc audio task implementation with optional visual cue
jsPsych.plugins['2afc_audio_visual'] = (function(){
    // preloads
    jsPsych.pluginAPI.registerPreload('2afc_audio_visual', 'stimulus', 'audio');
    jsPsych.pluginAPI.registerPreload('2afc_audio_visual', 'prime', 'image');

    // experimental setup
    const p_fixation = 300;
    const d_fixation = 50;
    const p_prime = 750;
    const d_prime = 250;
    const d_stimulus = 0;
    const p_choice = 3000;
    const d_choice = 0;
    const d_feedback = 750;

    // real-time rejection criterion
    var n_trials = 0;
    var n_misses = 0;
    var mhistory = [];
    const pct_criterion = 50;
    const min_criterion = 20;

    // language settings
    const lang = {
        'de': {
            'rejection': 'Leider muss das Experiment an dieser Stelle vorzeitig beendet werden, da die Zahl der verpassten Antworten zu hoch ist. Bei Fragen wenden Sie sich bitte an die Experimentleitung.'
        },
        'en': {
            'rejection': 'It looks like you are having difficulty with giving a timely response. Unfortunately, you cannot continue with the experiment due to the number of missed trials. Please direct any questions you may have to the experimenters.'
        }
    };

    // plugin specification
    var plugin = {};
    plugin.info = {
        name: '2afc_audio_visual',
        parameters: {
            study_language: {
                type: jsPsych.plugins.parameterType.STRING,
                default: 'de',
                pretty_name: 'Study language',
                description: 'Language settings for the study. Allowed options are: de, en'
            },
            no: {
                type: jsPsych.plugins.parameterType.INT,
                default: undefined,
                pretty_name: 'Number',
                description: 'Trial number.'
            },
            block: {
                type: jsPsych.plugins.parameterType.INT,
                default: undefined,
                pretty_name: 'Block number',
                description: 'Number of the block this trial is presented in.'
            },
            stimulus: {
                type: jsPsych.plugins.parameterType.AUDIO,
                default: undefined,
                pretty_name: 'Audio stimulus file',
                description: 'File to be played as the auditory stimulus of the trial.'
            },
            stimulus_path: {
                type: jsPsych.plugins.parameterType.STRING,
                default: './',
                pretty_name: 'Stimulus file path',
                description: 'Path to the stimulus file.'
            },
            prime: {
                type: jsPsych.plugins.parameterType.IMAGE,
                default: null,
                pretty_name: 'Visual cue file',
                description: 'File to be displayed as the visual prime of the trial (if any).'
            },
            prime_path: {
                type: jsPsych.plugins.parameterType.STRING,
                default: './',
                pretty_name: 'Prime file path',
                description: 'Path to the prime file.'
            },
            is_control: {
                type: jsPsych.plugins.parameterType.BOOL,
                default: false,
                pretty_name: 'Control stimulus flag',
                description: 'Flag (if true) for control stimuli.'
            },
            option_left: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: 'Left-hand side option',
                description: 'Option displayed on left-hand side of screen.'
            },
            option_right: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: 'Right-hand side option',
                description: 'Option displayed on right-hand side of screen.'
            },
            key_left: {
                type: jsPsych.plugins.parameterType.STRING,
                default: 'y',
                pretty_name: 'Left key',
                description: 'Key to accept for left answers.'
            },
            key_right: {
                type: jsPsych.plugins.parameterType.STRING,
                default: 'n',
                pretty_name: 'Right key',
                description: 'Key to accept for right answers.'
            },
            target_position: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: 'Target grid position',
                description: 'Position in the grid where the target is presented (left/right).'
            },
            enable_feedback: {
                type: jsPsych.plugins.parameterType.BOOL,
                default: false,
                pretty_name: 'Trial feedback',
                description: 'Whether or not feedback should be given after the participant\'s response.'
            },
            enable_prime: {
                type: jsPsych.plugins.parameterType.BOOL,
                default: false,
                pretty_name: 'Priming during trial',
                description: 'Whether or not a visual cue should be employed during the trial.'
            }
        }
    }

    // trial logic
    plugin.trial = function(e, t) {
        // log start of trial
        console.log("Trial", t, ": Starting...");

        // setup data structure
        let data = {
            no: t.no,
            block: t.block,
            stimulus: t.stimulus,
            prime: t.prime,
            with_prime: t.enable_prime,
            with_feedback : t.enable_feedback,
            is_control: t.is_control,
            option_left: t.option_left,
            option_right: t.option_right,
            target_position: t.target_position,
            choice_key: null,
            choice_option: null,
            choice_is_target: null,
            rt: null
        };

        // setup fixation
        let pFixation = function(){
            // display fixation cross
            let fixation = $(
                '<div id="trial-container">' + 
                    '<p align="center">+</p>' + 
                '</div>'
            );

            e.innerHTML = '';
            fixation.appendTo(e);
            
            // transition to delay
            jsPsych.pluginAPI.setTimeout(dFixation, p_fixation);
        }

        // setup post-fixation delay
        let dFixation = function(){
            // clear screen
            e.innerHTML = '';

            // transition to prime or stimulus
            jsPsych.pluginAPI.setTimeout((t.enable_prime && typeof t.prime !== 'undefined') ? pPrime : pStimulus, d_fixation);
        }

        // setup prime
        let pPrime = function(){
            console.log("test pPrime");

            // display prime
            let prime = $(
                '<div id="trial-container">' + 
                    '<p align="center">' + 
                        '<img id="prime" src="" />' + 
                    '</p>' + 
                '</div>'
            );

            let img = prime.find('#prime');
            e.innerHTML = '';
            prime.appendTo(e);
            img.attr('src', t.prime_path + "/" + t.prime);

            // transition to post-prime delay
            jsPsych.pluginAPI.setTimeout(dPrime, p_prime);
        }

        // setup post-prime delay
        let dPrime = function(){
            // clear screen
            e.innerHTML = '';

            // transition to stimulus
            jsPsych.pluginAPI.setTimeout(pStimulus, d_prime);
        }

        // setup audio stimulus
        let pStimulus = function(){
            // clear screen
            e.innerHTML = '';

            // setup audio context
            var context = jsPsych.pluginAPI.audioContext();
            
            if (context !== null) {
                var source = context.createBufferSource();
                source.buffer = jsPsych.pluginAPI.getAudioBuffer(t.stimulus);
                source.connect(context.destination);
            } else {
                var audio = jsPsych.pluginAPI.getAudioBuffer(t.stimulus);
                audio.currentTime = 0;
            }

            // setup audio ending
            var endAudio = function(){
                if (context !== null) {
                    source.stop();
                } else {
                    audio.pause();
                }
            }

            // setup listener removal
            var removeListener = function(){
                if (context !== null) {
                    source.onended = function(){};
                } else {
                    audio.removeEventListener('ended', transition);
                }
            }

            // setup transition
            var transition = function() {
                // stop audio & remove event listeners
                endAudio();
                removeListener();

                // transition to post-stimulus delay
                dStimulus();
            }

            // add event listener
            if (context !== null) {
                source.onended = transition;
            } else {
                audio.addEventListener('ended', transition);
            }

            // play stimulus
            if (context !== null) {
                source.start(context.currentTime);
            } else {
                audio.play();
            }
        }

        // setup post-audio delay
        let dStimulus = function(){
            // clear screen
            e.innerHTML = '';

            // transition to choice
            jsPsych.pluginAPI.setTimeout(pChoice, d_stimulus);
        }

        // setup choice
        let pChoice = function(){
            // setup feedback handler
            let pFeedback = function(){
                if (data.choice_is_target && data.choice_option == 'left') left.css('color', 'green');
                else if (!data.choice_is_target && data.choice_option == 'right') right.css('color', 'red');
                else if (data.choice_is_target && data.choice_option == 'right') right.css('color', 'green');
                else if (!data.choice_is_target && data.choice_option == 'left') left.css('color', 'red');

                jsPsych.pluginAPI.setTimeout(dChoice, d_feedback);
            }

            // setup key handler
            let keyHandler = function(event){
                if ([jsPsych.pluginAPI.convertKeyCharacterToKeyCode(t.key_left), jsPsych.pluginAPI.convertKeyCharacterToKeyCode(t.key_right)].includes(event.key)) {
                    jsPsych.pluginAPI.cancelKeyboardResponse(keyListener);
                    jsPsych.pluginAPI.clearAllTimeouts();

                    data.choice_key = event.key;
                    data.choice_option = (event.key == jsPsych.pluginAPI.convertKeyCharacterToKeyCode(t.key_left)) ? 'left' : 'right';
                    data.choice_is_target = (data.choice_option == t.target_position);
                    data.rt = event.rt;

                    if (t.enable_feedback) jsPsych.pluginAPI.setTimeout(pFeedback, d_choice);
                    else jsPsych.pluginAPI.setTimeout(dChoice, d_choice)
                }
            }

            // setup key listener
            let keyListener = jsPsych.pluginAPI.getKeyboardResponse({
                callback_function: keyHandler,
                valid_responses: jsPsych.ALL_KEYS,
                rt_method: 'performance',
                persist: true
            });

            // setup maximum rt timeout
            jsPsych.pluginAPI.setTimeout(function(){
                jsPsych.pluginAPI.cancelKeyboardResponse(keyListener);
                jsPsych.pluginAPI.clearAllTimeouts();

                n_misses += 1;
                mhistory.push(data.no);
                data.choice_key = null;
                data.choice_option = null;
                data.choice_is_target = false;
                data.rt = null;

                jsPsych.pluginAPI.setTimeout(dChoice, d_choice);
            }, p_choice);

            // display choices
            let choice = $(
                '<div id="trial-container">' + 
                    '<p align="center">' + 
                        '<span id="opt-left" class="opt"></span>&nbsp;&nbsp;<span class="opt-divider">|</span>&nbsp;&nbsp;<span id="opt-right" class="opt"></span>' + 
                    '</p>' + 
                '</div>'
            );

            let left = choice.find('#opt-left');
            let right = choice.find('#opt-right');

            e.innerHTML = '';
            choice.appendTo(e);
            left.html(t.option_left);
            right.html(t.option_right);
        }

        // rtc logic
        let participant_is_attending = function() {
            let missed_in_last_n = 0;

            for (let i = 0; i < mhistory.length; i++) {
                if (mhistory[i] >= (data.no - min_criterion) && mhistory[i] >= 0) {
                    missed_in_last_n += 1;
                }
            }

            return missed_in_last_n <= (min_criterion * (pct_criterion / 100));
        }

        // setup post-choice delay
        let dChoice = function(){
            // clear screen
            e.innerHTML = '';
            console.log("Trial", t, ": Completed!");

            // end trial, if real-time criterion looks alright
            //if ((n_misses / n_trials * 100) <= pct_criterion || n_trials < min_criterion) {
            if (participant_is_attending()) {
                jsPsych.pluginAPI.cancelAllKeyboardResponses();
                jsPsych.pluginAPI.clearAllTimeouts();
                jsPsych.finishTrial(data);
            } else {
                // otherwise, abort experiment
                console.log("Experiment aborted due to excessive number of missed responses.");

                jsPsych.pluginAPI.clearAllTimeouts();
                jsPsych.pluginAPI.cancelAllKeyboardResponses();

                let rejection = $(
                    '<div id="trial-container">' + 
                        '<p align="center">' + lang[t.study_language].rejection + '</p>' + 
                    '</div>'
                );
                rejection.appendTo(e);
            }
            
        }

        // start trial
        n_trials += 1;
        pFixation();
    }

    return plugin;
})();