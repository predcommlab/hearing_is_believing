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
    const p_fixation = 500;
    const d_fixation = 50;
    const p_prime = 750;
    const d_prime = 0;
    const d_stimulus = 500;
    const p_choice = 2000;
    const d_choice = 0;
    const d_feedback = 750;
    const prime_max_height = 400;

    // real-time rejection criterion
    var n_trials = 0;
    var n_misses = 0;
    var mhistory = [];
    const pct_criterion = 50;
    const min_criterion = 20;

    // language settings
    const lang = {
        'de': {
            'rejection': 'Leider muss das Experiment an dieser Stelle vorzeitig beendet werden, da die Zahl der verpassten Antworten zu hoch ist. Bei Fragen wenden Sie sich bitte an die Experimentleitung.',
            'miss': 'Zu langsam!'
        },
        'en': {
            'rejection': 'It looks like you are having difficulty with giving a timely response. Unfortunately, you cannot continue with the experiment due to the number of missed trials. Please direct any questions you may have to the experimenters.',
            'miss': 'Too slow!'
        }
    };

    // plugin specification
    var plugin = {};
    plugin.n_trials = 0;
    plugin.n_misses = 0;
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
            prime: {
                type: jsPsych.plugins.parameterType.IMAGE,
                default: null,
                pretty_name: 'Visual cue file',
                description: 'File to be displayed as the visual prime of the trial (if any).'
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
            },
            enable_persistence: {
                type: jsPsych.plugins.parameterType.BOOL,
                default: false,
                pretty_name: 'Prime persistence',
                description: 'Should the prime be displayed through-out audio presentation?'
            },
            enable_persistence_centred: {
                type: jsPsych.plugins.parameterType.BOOL,
                default: false,
                pretty_name: 'Persistence centred',
                description: 'Should options be presented centred on top of the prime (e.g., to avoid predictive eye movements towards the bottom)?'
            },
            rtc_callback: {
                type: jsPsych.plugins.parameterType.FUNCTION,
                default: function(e){},
                pretty_name: 'RTC callback',
                description: 'Function for the callback in case the RTC is negative (i.e., participant should be excluded).'
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
            with_persistence: t.enable_persistence,
            with_persistence_centred: t.enable_persistence_centred,
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
                    '<div id="fix-container" style="position: relative !important; height: {0}px;">'.format(prime_max_height.toString()) + 
                        '<p align="center" style="font-size: 30px; font-weight: normal; position: absolute !important; left: 50%; top: 35%; transform: translate(-50%, -35%);">' +
                            '+' + 
                        '</p>' + 
                    '</div>' + 
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
            // display prime
            let prime = $(
                '<div id="trial-container">' + 
                    '<div align="center" id="img-container" style="position: relative !important;">' + 
                        '<img id="prime" src="" style="border: 5px solid white; border-radius: 40px; max-height: {0}px; height: auto;" />'.format(prime_max_height.toString()) + 
                        (t.enable_persistence ? (!t.enable_persistence_centred ? '<p id="trial-container-padded">&nbsp;</p>' : '<p id="trial-container-padded" style="position: absolute !important; left: 50%; top: 40%; transform: translate(-50%, -40%); background-color: rgba(255, 255, 255, 0.65); border-radius: 8px;"></p>') : '') + 
                    '</div>' + 
                '</div>'
            );

            let img = prime.find('#prime');
            e.innerHTML = '';
            prime.appendTo(e);
            img.attr('src', t.prime);

            // transition to post-prime delay
            jsPsych.pluginAPI.setTimeout(dPrime, p_prime);
        }

        // setup post-prime delay
        let dPrime = function(){
            // clear screen
            if (!t.enable_persistence) {
                e.innerHTML = '';
            }

            // transition to stimulus
            jsPsych.pluginAPI.setTimeout(pStimulus, d_prime);
        }

        // setup audio stimulus
        let pStimulus = function(){
            // clear screen
            if (!t.enable_persistence) {
                e.innerHTML = '';
            }

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
            if (!t.enable_persistence) { e.innerHTML = ''; }

            // transition to choice
            jsPsych.pluginAPI.setTimeout(pChoice, d_stimulus);
        }

        // setup choice
        let pChoice = function(){
            // setup feedback handler
            let pFeedback = function(){
                /*
                // adjusted for is_control flag reverting the feedback'd target while maintaining
                // the target coding for analyses down the line

                if ((data.choice_is_target && data.choice_option == 'left' && !data.is_control) || 
                    (!data.choice_is_target && data.choice_option == 'left' && data.is_control)) {
                    left.css('color', 'green');
                } else if ((!data.choice_is_target && data.choice_option == 'left' && !data.is_control) ||
                           (data.choice_is_target && data.choice_option == 'left' && data.is_control)) {
                    left.css('color', 'red')
                } else if ((data.choice_is_target && data.choice_option == 'right' && !data.is_control) ||
                           (!data.choice_is_target && data.choice_option == 'right' && data.is_control)) {
                    right.css('color', 'green');
                } else if ((!data.choice_is_target && data.choice_option == 'right' && !data.is_control) || 
                           (data.choice_is_target && data.choice_option == 'right' && data.is_control)) {
                    right.css('color', 'red');
                } else if ((!data.choice_is_target && data.choice_option == null && t.target_position == 'left' && !data.is_control) ||
                           (!data.choice_is_target && data.choice_option == null && t.target_position == 'right' && data.is_control)) {
                    left.css('color', 'green');
                } else if ((!data.choice_is_target && data.choice_option == null && t.target_position == 'right' && !data.is_control) ||
                           (!data.choice_is_target && data.choice_option == null && t.target_position == 'left' && data.is_control)) {
                    right.css('color', 'green');
                }
                */

                // adjusted for redesigned task where is_control does not
                // require a reverted coding

                if (data.choice_is_target && data.choice_option == 'left') {
                    left.css('color', 'green');
                    //$('#prime').css('border', '5px solid green');
                } else if (data.choice_is_target && data.choice_option == 'right') {
                    right.css('color', 'green');
                    //$('#prime').css('border', '5px solid green');
                } else if (!data.choice_is_target && data.choice_option == 'left') {
                    left.css('color', 'red');
                    //$('#prime').css('border', '5px solid red');
                } else if (!data.choice_is_target && data.choice_option == 'right') {
                    right.css('color', 'red');
                    //$('#prime').css('border', '5px solid red');
                } else if (!data.choice_is_target && data.choice_option == null && t.target_position == 'left') {
                    left.css('color', 'green');
                    //$('#prime').css('border', '5px solid green');
                } else if (!data.choice_is_target && data.choice_option == null && t.target_position == 'right') {
                    right.css('color', 'green');
                    //$('#prime').css('border', '5px solid green');
                }

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

                $(e).find('#miss-msg').show();

                n_misses += 1;
                plugin.n_misses += 1;
                mhistory.push(data.no);
                data.choice_key = null;
                data.choice_option = null;
                data.choice_is_target = false;
                data.rt = null;

                if (t.enable_feedback) jsPsych.pluginAPI.setTimeout(pFeedback, d_choice);
                else jsPsych.pluginAPI.setTimeout(dChoice, d_choice);
            }, p_choice);
            
            
            if (t.enable_persistence) {
                if (!t.enable_persistence_centred) {
                    // display choices on bottom
                    var choice = $(
                        '<span id="opt-left" class="opt"></span>&nbsp;&nbsp;&nbsp;<span class="opt-divider">|</span>&nbsp;&nbsp;&nbsp;<span id="opt-right" class="opt"></span>'
                    );
                } else {
                    // display choices centred
                    var choice = $(
                        '<span>&nbsp;&nbsp;&nbsp;<span id="opt-left" class="opt"></span>&nbsp;&nbsp;&nbsp;<span class="opt-divider">|</span>&nbsp;&nbsp;&nbsp;<span id="opt-right" class="opt"></span>&nbsp;&nbsp;&nbsp;</span>'
                    );
                }

                let miss = $(
                    '<span id="miss-msg">{0}<br/></span>'.format(lang[t.study_language].miss)
                );

                let trialContainerP = $(e).find('#trial-container-padded');
                trialContainerP.html('');
                miss.appendTo(trialContainerP);
                $(e).find('#miss-msg').hide();
                choice.appendTo(trialContainerP);
                
                var left = $(e).find('#opt-left');
                var right = $(e).find('#opt-right');
                let padding_left = '&nbsp;'.repeat(t.option_left.length < t.option_right.length ? t.option_right.length - t.option_left.length : 0);
                let padding_right = '&nbsp;'.repeat(t.option_left.length > t.option_right.length ? t.option_left.length - t.option_right.length : 0);
                
                left.html(padding_left + t.option_left);
                right.html(t.option_right + padding_right);
            } else {
                // display choices
                var choice = $(
                    '<div id="trial-container">' + 
                        '<p align="center">' + 
                            '<span id="opt-left" class="opt"></span>&nbsp;&nbsp;<span class="opt-divider">|</span>&nbsp;&nbsp;<span id="opt-right" class="opt"></span>' + 
                        '</p>' + 
                    '</div>'
                );

                var left = choice.find('#opt-left');
                var right = choice.find('#opt-right');
                let padding_left = '&nbsp;'.repeat(t.option_left.length < t.option_right.length ? t.option_right.length - t.option_left.length : 0);
                let padding_right = '&nbsp;'.repeat(t.option_left.length > t.option_right.length ? t.option_left.length - t.option_right.length : 0);

                e.innerHTML = '';
                choice.appendTo(e);
                left.html(padding_left + t.option_left);
                right.html(t.option_right + padding_right);
            }
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

                if (typeof(t.rtc_callback) !== 'undefined') {
                    t.rtc_callback(e);
                }
            }
            
        }

        // start trial
        n_trials += 1;
        plugin.n_trials += 1;
        pFixation();
    }

    return plugin;
})();