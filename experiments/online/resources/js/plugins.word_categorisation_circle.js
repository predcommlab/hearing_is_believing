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

// word categorisation in a circle of images plugin
jsPsych.plugins['word_categorisation_circle'] = (function(){
    // trial settings
    const p_fixation = 300;
    const d_fixation = 50;

    // plugin specification
    var plugin = {};
    plugin.info = {
        name: 'word_categorisation_circle',
        parameters: {
            no: {
                type: jsPsych.plugins.parameterType.INT,
                default: 0,
                pretty_name: 'Number',
                description: 'Trial number.'
            },
            block: {
                type: jsPsych.plugins.parameterType.INT,
                default: 0,
                pretty_name: 'Block',
                description: 'Block number.'
            },
            images: {
                type: jsPsych.plugins.parameterType.COMPLEX,
                default: [],
                pretty_name: 'Image files',
                description: 'Image files to be categorised by participant.'
            },
            stimulus: {
                type: jsPsych.plugins.parameterType.STRING,
                default: undefined,
                pretty_name: 'Stimulus',
                decsription: 'The word (in plain text/html) to be categorised.'
            },
            expectation: {
                type: jsPsych.plugins.parameterType.STRING,
                default: '',
                pretty_name: 'Expected image',
                description: 'The expected image that the participant selects (if any).'
            },
            enable_fixation: {
                type: jsPsych.plugins.parameterType.BOOL,
                default: true,
                pretty_name: 'Enable fixation',
                description: 'Should a fixation cross be displayed before trials?'
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
            images: t.images.join(),
            stimulus: t.stimulus,
            expectation: t.expectation,
            enable_fixation: t.enable_fixation,
            choice: null,
            correct: null,
            rt: null
        };

        // clear canvas
        e.innerHTML = '';
        
        // fixation logic
        let pFixation = function(){
            // clear container
            e.innerHTML = '';

            // create fixation cross
            let fixationContainer = $(
                '<div id="fixation-container" align="center">' + 
                    '+' + 
                '</div>'
            );

            // show fixation cross, move on
            fixationContainer.appendTo(e);
            jsPsych.pluginAPI.setTimeout(() => {jsPsych.pluginAPI.setTimeout(pTrial, d_fixation)}, p_fixation);
        };

        // trial logic
        let pTrial = function(){
            // clear container
            e.innerHTML = '';

            // window setup
            const h = window.innerHeight;
            const w = window.innerWidth;

            // scale circle accordingly
            const p = Math.min(h/5, w/5)
            const d = 0.65 * Math.min(h-p, w-p);
            const r = d/2;
            const Z = [h/2, w/2];
            const delta = 2*Math.PI / t.images.length;
            let theta = 2*Math.PI;

            // create trial container
            let trialContainer = $(
                '<fieldset style="border: none;">' + 
                    '<div id="image-container" style="width: 100% !important; height: 100% !important;" align="center">' + 
                        Object.entries(t.images).reduce((str, [k, v]) => {
                            const x = Math.round(Z[1] + r*Math.cos(theta));
                            const y = Math.round(Z[0] + r*Math.sin(theta));

                            theta -= delta;

                            return str + 
                                '<a href="javascript:void(0);">' + 
                                        '<img src="{0}" style="max-height: {1}px; max-width: {2}px; height: auto; width: auto; left: {3}px; top: {4}px; position: absolute !important;" data-ref="{5}" />'.format(v, p.toString(), p.toString(), x.toString(), y.toString(), k.toString()) + 
                                '</a>';
                        }, '') +
                        '<div align="center">' + t.stimulus + '</div>' + 
                    '</div>' + 
                '</fieldset>'
            );

            // grab relevant elements
            let imageButtons = trialContainer.find('img');

            // setup element recentering
            imageButtons.on('load', function(){
                let x = $(this).css('left'); x = parseInt(x.substr(0, x.length - 2));
                let y = $(this).css('top'); y = parseInt(y.substr(0, y.length - 2));
                
                dx = x - this.width / 2;
                dy = y - this.height / 2;

                $(this).css('left', dx);
                $(this).css('top', dy);
            });

            // handle clicks
            imageButtons.on('click', function(){
                data.rt = performance.now() - time_start;
                data.choice = $(this).attr('src');
                data.correct = t.expectation.length > 0 ? data.choice == t.expectation : null;

                imageButtons.off('load');
                imageButtons.off('click');
                jsPsych.pluginAPI.clearAllTimeouts();
                jsPsych.finishTrial(data);
            });

            // show trial, start RT measurement
            trialContainer.appendTo(e);
            const time_start = performance.now();
        };

        // start trial
        if (t.enable_fixation) {
            pFixation();
        } else {
            pTrial();
        }
    }

    return plugin;
})();