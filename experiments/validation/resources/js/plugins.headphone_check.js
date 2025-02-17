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

// quick and dirty implementation of the headphone check detaield in:
//      Woods, K.J.P., Siegel, M.H., Traer, J., & McDermott, J.H. (2017). Headphone screening to facilitate web-based auditory experiments. Attention, Perception & Psychophysics. 10.3758/s13414-017-1361-2
jsPsych.plugins['headphone_check'] = (function(){
    // preloads
    jsPsych.pluginAPI.registerPreload('headphone_check', 'file_calibration', 'audio');
    jsPsych.pluginAPI.registerPreload('headphone_check', 'file_stim1', 'audio');
    jsPsych.pluginAPI.registerPreload('headphone_check', 'file_stim2', 'audio');
    jsPsych.pluginAPI.registerPreload('headphone_check', 'file_stim3', 'audio');
    jsPsych.pluginAPI.registerPreload('headphone_check', 'file_stim4', 'audio');
    jsPsych.pluginAPI.registerPreload('headphone_check', 'file_stim5', 'audio');
    jsPsych.pluginAPI.registerPreload('headphone_check', 'file_stim6', 'audio');

    // experimental setup
    const p_fixation = 300;
    const d_fixation = 50;
    const d_stimulus = 0;
    const p_choice = 5000;
    const d_choice = 500;

    // language settings
    const lang = {
        'de': {
            // navigation text
            'navigation_continue': 'Weiter',
            'navigation_previous': 'Zurück',

            // calibration text
            'instructions_header': 'Kalibrierung der Kopfhörer',
            'instructions_body': 'Bitte beachten Sie, dass diese Studie <b>ausschließlich</b> mit Kopfhörern abgeschloßen werden kann. ' + 
                                 'Falls Sie derzeit keine Kopfhörer mit Ihrem Computer verbunden haben, tun Sie dies bitte bevor Sie die unten angeführten Schritte befolgen.' + 
                                 '<br /><br />Wenn Sie Ihre Kopfhörer verbunden haben, sollte die Audiowiedergabe kalibriert werden. ' + 
                                 'Dazu stellen Sie bitte die Systemlautstärke auf ungefähr 25% des Maximums. ' + 
                                 'Drücken Sie danach auf `Abspielen`, um ein Probegeräusch zu hören. ' + 
                                 'Nun stellen Sie bitte die Systemlautstärke so ein, dass das Geräusch laut aber komfortabel zu hören ist. ' + 
                                 'Dazu können Sie das Rauschen so oft wie nötig neu abspielen.' + 
                                 '<br /><br />Sobald Sie eine gute Lautstärke gefunden haben, drücken Sie bitte auf `Weiter`. Bitte beachten Sie, dass diese Option erst dann erscheint, wenn Sie das Probegeräusch abgespielt haben.',
            'instructions_play': 'Abspielen',

            // trial text
            'instructions2_header': 'Test der Kopfhörer',
            'instructions2_body': 'Im folgenden Teil werden Sie {0} Tonfolgen hören. ' + 
                                  'Diese bestehen jeweils aus drei Tönen, wobei jede Tonfolge genau einen Ton beeinhaltet, der deutlich leiser abgespielt wird als die anderen beiden.' + 
                                  '<br /><br />Ihre Aufgabe ist es, diesen leiseren Ton zu identifizieren. ' + 
                                  'Dazu nutzen Sie bitte die Tasten `1`, `2` oder `3`, die auf Ihrer Tastatur über den Buchstaben liegen. ' + 
                                  'Wenn Sie also beispielsweise eine Tonfolge hören, in der der zweite Ton leiser als der erste und dritte war, dann würden Sie die Taste `2` drücken. ' + 
                                  'Sie haben jeweils fünf Sekunden Zeit, um Ihre Antwort zu geben.' + 
                                  '<br /><br />Sobald Sie bereit sind, mit dieser Aufgabe zu starten, drücken Sie bitte auf `Weiter`, um den Kopfhörertest zu beginnen. ' + 
                                  'Sollten Sie Ihre Kopfhörer erneut kalibrieren wollen, drücken Sie nun stattdessen auf `Zurück`.',
            'rejection': 'Vergewissern Sie sich bitte, dass Sie unbeschädigte Kopfhörer benutzen. Leider kann das Experiment nicht fortgesetzt werden. Bei Fragen richten Sie sich ggf. bitte über Prolific an die Experimentleitung.',
            'success': 'Klasse! Sie haben den Kopfhörertest erfolgreich beendet. Sie können nun zum Hauptexperiment übergehen. Drücken Sie bitte auf `Weiter`, um fortzufahren.'
        }
    };

    // plugin specification
    var plugin = {};
    plugin.info = {
        name: 'headphone_check',
        parameters: {
            study_language: {
                type: jsPsych.plugins.parameterType.STRING,
                default: 'de',
                pretty_name: 'Study language',
                description: 'The language to be used for the study.'
            },
            study_title: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: "Study title",
                description: "Title of the study, to be displayed in the brief."
            },
            study_subtitle: {
                type: jsPsych.plugins.parameterType.STRING,
                default: "",
                pretty_name: "Study subtitle",
                description: "Subtitle of the study, to be displayed in the brief."
            },
            no: {
                type: jsPsych.plugins.parameterType.INT,
                default: 6,
                pretty_name: 'Number',
                description: 'The number of trials to be played.'
            },
            threshold: {
                type: jsPsych.plugins.parameterType.FLOAT,
                default: 5/6,
                pretty_name: 'Threshold',
                description: 'The threshold criterion that participants must score to continue.'
            },
            file_calibration: {
                type: jsPsych.plugins.parameterType.AUDIO,
                default: './resources/audio/noise_calib_stim.wav',
                pretty_name: 'Calibration audio file',
                description: 'Audio file used for volume calibration.'
            },
            file_stim1: {
                type: jsPsych.plugins.parameterType.AUDIO,
                default: './resources/audio/antiphase_HC_IOS.wav',
                pretty_name: 'Stimulus one',
                description: 'Audio file of stimulus one.'
            },
            corr_stim1: {
                type: jsPsych.plugins.parameterType.INT,
                default: 3,
                pretty_name: "Correct value stimulus one",
                description: "Value to be chosen for stimulus one."
            },
            file_stim2: {
                type: jsPsych.plugins.parameterType.AUDIO,
                default: './resources/audio/antiphase_HC_ISO.wav',
                pretty_name: 'Stimulus two',
                description: 'Audio file of stimulus two.'
            },
            corr_stim2: {
                type: jsPsych.plugins.parameterType.INT,
                default: 2,
                pretty_name: "Correct value stimulus two",
                description: "Value to be chosen for stimulus two."
            },
            file_stim3: {
                type: jsPsych.plugins.parameterType.AUDIO,
                default: './resources/audio/antiphase_HC_OIS.wav',
                pretty_name: 'Stimulus three',
                description: 'Audio file of stimulus three.'
            },
            corr_stim3: {
                type: jsPsych.plugins.parameterType.INT,
                default: 3,
                pretty_name: "Correct value stimulus three",
                description: "Value to be chosen for stimulus three."
            },
            file_stim4: {
                type: jsPsych.plugins.parameterType.AUDIO,
                default: './resources/audio/antiphase_HC_OSI.wav',
                pretty_name: 'Stimulus four',
                description: 'Audio file of stimulus four.'
            },
            corr_stim4: {
                type: jsPsych.plugins.parameterType.INT,
                default: 2,
                pretty_name: "Correct value stimulus four",
                description: "Value to be chosen for stimulus four."
            },
            file_stim5: {
                type: jsPsych.plugins.parameterType.AUDIO,
                default: './resources/audio/antiphase_HC_SIO.wav',
                pretty_name: 'Stimulus five',
                description: 'Audio file of stimulus five.'
            },
            corr_stim5: {
                type: jsPsych.plugins.parameterType.INT,
                default: 1,
                pretty_name: "Correct value stimulus five",
                description: "Value to be chosen for stimulus five."
            },
            file_stim6: {
                type: jsPsych.plugins.parameterType.AUDIO,
                default: './resources/audio/antiphase_HC_SOI.wav',
                pretty_name: 'Stimulus six',
                description: 'Audio file of stimulus six.'
            },
            corr_stim6: {
                type: jsPsych.plugins.parameterType.INT,
                default: 1,
                pretty_name: "Correct value stimulus six",
                description: "Value to be chosen for stimulus six."
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
            threshold: t.threshold,
            accuracy: null,
        };

        // setup trial container
        let trialContainer = $(
            '<div id="trial-header" style="width: 100%; font-size: small;" align="center">' + 
                '<div id="trial-header-line1" style="width: 75%; display: flex; justify-content: space-between;" align="justify">' + 
                    '<p><b>' + lang[t.study_language].instructions_header + '</b></p>' + 
                    '<p><i>' + t.study_title + '</i></p>' + 
                '</div>' + 
                '<div id="trial-header-line2" style="width: 75%; display: flex; justify-content: space-between;" align="justify">' + 
                    '<p><b>' + lang[t.study_language].instructions2_header + '</b></p>' + 
                    '<p><i>' + t.study_title + '</i></p>' + 
                '</div>' + 
                '<div style="width: 75%;" align="justify">' + t.study_subtitle + '</div>' + 
            '</div>' + 
            '<div id="trial-super-container" style="width: 100%; max-height:70vh; overflow-y:scroll; font-size: small; background: #eee;" align="center">' +
                '<div id="trial-container" style="width: 75%;" align="justify">' + 
                    '<div id="trial-instructions"></div>' + 
                    '<div id="trial-instructions2"></div>' + 
                    '<div id="trial-rejection"></div>' + 
                    '<div id="trial-success">' + 
                        lang[t.study_language].success + 
                        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;' + 
                        '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;' +
                    '</div>' + 
                '</div>' + 
            '</div>' + 
            '<div id="trial-main-container">' + 
                '<div id="trial-counter"></div>' + 
                '<div id="trial-fixation"></div>' + 
                '<div id="trial-options"></div>' + 
            '</div>' + 
            '<formset>' + 
                '<button id="trial-previous" type="button">' + lang[t.study_language].navigation_previous + '</button>' + 
                '&nbsp;&nbsp;&nbsp;' + 
                '<button id="trial-continue" type="button">' + lang[t.study_language].navigation_continue + '</button>' + 
            '</formset>'
        );

        // get headers
        let headerInstructions = trialContainer.find('#trial-header-line1');
        let headerInstructions2 = trialContainer.find('#trial-header-line2');

        // grab success
        let trialSuccess = trialContainer.find('#trial-success');
        
        // setup trial instructions
        let trialInstructions = $(
            '<p>' + lang[t.study_language].instructions_body + '</p>' + 
            '<fieldset>' + 
                '<button id="button-calibration" type="button">' + lang[t.study_language].instructions_play + '</button>' + 
            '</fieldset><br />'
        );

        // setup trial instructions2
        let trialInstructions2 = $(
            '<p>' + lang[t.study_language].instructions2_body.format(t.no.toString()) + '</p>'
        );

        // setup rejection screen
        let trialRejection = $(
            '<p>' + lang[t.study_language].rejection + '</p>'
        );

        // setup fixation cross
        let trialFixation = $(
            '<span>+</span>'
        );
        
        // setup counter
        let trialCounter = $(
            '<span>0%</span>'
        );

        // setup options
        let trialOptions = $(
            '<span>1&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;3</span>'
        );

        // grab buttons
        let buttonCalibration = trialInstructions.find('#button-calibration');
        let buttonStart = trialInstructions2.find('#button-start');
        let buttonContinue = trialContainer.find('#trial-continue');
        let buttonPrevious = trialContainer.find('#trial-previous');

        // setup audio context
        let context = jsPsych.pluginAPI.audioContext();
        let isPlaying = false;

        // setup audio play
        let playAudio = function(f, cb) {
            // make sure audio is off
            if (isPlaying == true) {
                return;
            }
            
            // connect source, setup callback & play sound
            if (context !== null) {
                var source = context.createBufferSource();
                source.buffer = jsPsych.pluginAPI.getAudioBuffer(f);
                source.connect(context.destination);
                
                source.onended = function(){
                    isPlaying = false;
                    source.onended = function(){};
                    cb();
                }
                
                source.start(context.currentTime);
            } else {
                var audio = jsPsych.pluginAPI.getAudioBuffer(f);
                audio.currentTime = 0;

                let callback = function(){
                    isPlaying = false;
                    audio.removeEventListener('ended', callback);
                    cb();
                }
                audio.addEventListener('ended', callback);
                
                audio.play();
            }
        }
        
        // add containers
        trialInstructions.appendTo(trialContainer.find('#trial-instructions'));
        trialInstructions2.appendTo(trialContainer.find('#trial-instructions2'));
        trialRejection.appendTo(trialContainer.find('#trial-rejection'));
        trialFixation.appendTo(trialContainer.find('#trial-fixation'));
        trialCounter.appendTo(trialContainer.find('#trial-counter'));
        trialOptions.appendTo(trialContainer.find('#trial-options'));
        
        // setup clean up
        let clearAll = function(){
            // toggle invisibility
            headerInstructions.hide();
            headerInstructions2.hide();
            trialInstructions.hide();
            trialInstructions2.hide();
            trialRejection.hide();
            trialSuccess.hide();
            trialFixation.hide();
            trialCounter.hide();
            trialOptions.hide();
            buttonContinue.hide();
            buttonPrevious.hide();

            // remove button event listeners
            buttonContinue.off('click');
            buttonPrevious.off('click');
            buttonCalibration.off('click');
            buttonStart.off('click');
        }

        // setup calibration
        let pCalibration = function(){
            // clean up
            clearAll();

            // show relevant sections
            headerInstructions.show();
            trialInstructions.show();
            
            // add event listeners
            buttonCalibration.on('click', function(){
                playAudio(t.file_calibration, function(){
                    buttonContinue.show();
                });
            });
            buttonContinue.on('click', pTrialInstructions);
        };

        // setup trial instructions
        let pTrialInstructions = function(){
            // clean up
            clearAll();

            // show relevant sections
            headerInstructions2.show();
            trialInstructions2.show();
            buttonPrevious.show();
            buttonContinue.show();

            // add event listeners
            buttonPrevious.on('click', pCalibration);
            buttonContinue.on('click', pTrials);
        }

        // setup trials
        let pTrials = async function(){
            // clean up
            clearAll();
            $('#trial-header').hide();

            // setup trial structures
            let files = [t.file_stim1, t.file_stim2, t.file_stim3, t.file_stim4, t.file_stim5, t.file_stim6];
            let answers = [t.corr_stim1, t.corr_stim2, t.corr_stim3, t.corr_stim4, t.corr_stim5, t.corr_stim6];
            let correct = 0;
            
            // setup current pool of options
            let options = [];

            // loop over requested number of trials
            for (let i = 0; i < t.no; i++) {
                // restock options if empty
                if (options.length == 0) {
                    options = [...Array(t.no).keys()];
                }

                // sample and consume option
                let indx = Math.floor(Math.random() * options.length);
                let current = options[indx];
                options.splice(indx, 1);

                // run single trial
                let outcome = await pSingleTrial(files[current], answers[current])
                              .then((corr) => { return corr; });
                correct += outcome;
            }

            // clean up
            clearAll();

            // show relevant sections
            headerInstructions2.show();
            $('#trial-header').show();

            // save accuracy
            data.accuracy = correct / t.no;

            // check performance
            if (correct / t.no < t.threshold) {
                // lock participant iff performance is poor
                trialRejection.show();
            } else {
                // otherwise, show final screen & finish up
                trialSuccess.show();
                buttonContinue.show();

                // setup event handler
                buttonContinue.on('click', function(){
                    jsPsych.finishTrial(data);
                });
            }
        };

        // setup async single trial presentation
        let pSingleTrial = function(stimulus, correct_opt){
            // setup fixation
            let pFixation = async function(){
                // clean up
                clearAll();

                // show relevant sections
                trialFixation.show();

                // pass on and return promise
                return new Promise(async (resolve, reject) => {
                    setTimeout(async () => {
                        await dFixation()
                              .then((corr) => {
                                resolve(corr);
                              });
                    }, p_fixation)
                })
            };

            // setup fixation delay
            let dFixation = async function(){
                // clean up
                clearAll();

                // pass on and return promise
                return new Promise(async (resolve, reject) => {
                    setTimeout(async () => {
                        await pAudio()
                              .then((corr) => {
                                resolve(corr);
                              });
                    }, d_fixation);
                });
            };

            // setup audio presentation
            let pAudio = async function(){
                // clean up
                clearAll();

                // play sound
                playAudio(stimulus, () => {});

                // pass on and return promise
                return new Promise(async (resolve, reject) => {
                    setTimeout(async () => {
                        await pChoice()
                              .then((corr) => {
                                resolve(corr);
                              });
                    }, 4000 + d_stimulus);
                });
            };

            // setup choice presentation
            let pChoice = async function(){
                // clean up
                clearAll();

                // show relevant sections
                trialOptions.show();
                
                // return promise (that handles input)
                return new Promise((resolve, reject) => {
                    // setup the key handler
                    let keyHandler = function(event) {
                        // allowed options
                        let keys = [jsPsych.pluginAPI.convertKeyCharacterToKeyCode('1'),
                                    jsPsych.pluginAPI.convertKeyCharacterToKeyCode('2'),
                                    jsPsych.pluginAPI.convertKeyCharacterToKeyCode('3')];
                        
                        // continue iff valid key
                        if (!keys.includes(event.key)) {
                            return;
                        }

                        // cancel response and timeout
                        jsPsych.pluginAPI.cancelKeyboardResponse(keyListener);
                        jsPsych.pluginAPI.clearAllTimeouts();

                        // is it corrrect?
                        let is_correct = correct_opt == (keys.indexOf(event.key) + 1);

                        // clean up
                        clearAll();

                        // resolve and return
                        setTimeout(() => {
                            resolve(is_correct);
                        }, d_choice);
                    }

                    // wait for keyboard response
                    let keyListener = jsPsych.pluginAPI.getKeyboardResponse({
                        callback_function: keyHandler,
                        valid_responses: jsPsych.ALL_KEYS,
                        rt_method: 'performance',
                        persist: true
                    });

                    // setup response maximum duration
                    jsPsych.pluginAPI.setTimeout(function(){
                        // clean up
                        clearAll();

                        // cancel response
                        jsPsych.pluginAPI.cancelKeyboardResponse(keyHandler);

                        // make trial false
                        let is_correct = false;

                        // resolve and return
                        setTimeout(() => {
                            resolve(is_correct);
                        }, d_choice);

                    }, p_choice);
                });
            };

            // start the chain of promises
            return new Promise(async (resolve, reject) => {
                let is_correct = await pFixation()
                                 .then((corr) => {
                                    resolve(corr);
                                    return corr;
                                  });
                return is_correct;
            });
        }
        
        // setup canvas
        e.innerHTML = '';
        trialContainer.appendTo(e);

        // start process
        pCalibration();
    }

    return plugin;
})();