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

// jQuery.shake
// as per phpslightly on https://stackoverflow.com/questions/4399005/implementing-jquerys-shake-effect-with-animate (last accessed: 30/01/2023 18:59)
jQuery.fn.shake = function(interval,distance,times){
    interval = typeof interval == "undefined" ? 100 : interval;
    distance = typeof distance == "undefined" ? 10 : distance;
    times = typeof times == "undefined" ? 3 : times;
    var jTarget = $(this);
    jTarget.css('position','relative');
    for(var iter=0;iter<(times+1);iter++){
       jTarget.animate({ left: ((iter%2==0 ? distance : distance*-1))}, interval);
    }
    return jTarget.animate({ left: 0},interval);
 }

// image categorisation plugin
jsPsych.plugins['image_categorisation'] = (function(){
    // setup
    const image_max_height = 250;

    // language settings
    const lang = {
        'de': {
            'instructions': 'Wählen Sie bitte für jedes Bild die richtige Kategorie aus.',
            'instructions_exclusive': 'Wählen Sie bitte für jedes Bild die richtige Kategorie aus. <b>Achtung!</b> Jede Kategorie kann nur einmal zugewiesen werden.',
            'please_select': 'Bitte auswählen...',
            'submit': 'Weiter',
            'reset': 'Zurücksetzen'
        },
        'en': {
            'instructions': 'Please select the correct category label for all images.',
            'instructions_exclusive': 'Please select the correct category label for all images. <b>Note</b> that each category may be assigned only once.',
            'please_select': 'Make a selection...',
            'submit': 'Submit',
            'reset': 'Reset'
        }
    };

    // plugin specification
    var plugin = {};
    plugin.info = {
        name: 'image_categorisation',
        parameters: {
            study_language: {
                type: jsPsych.plugins.parameterType.STRING,
                default: 'de',
                pretty_name: 'Study language',
                description: 'Language settings for the study. Allowed options are: de, en'
            },
            no: {
                type: jsPsych.plugins.parameterType.INT,
                default: 0,
                pretty_name: 'Number',
                description: 'Trial number.'
            },
            images: {
                type: jsPsych.plugins.parameterType.COMPLEX,
                default: [],
                pretty_name: 'Image files',
                description: 'Image files to be categorised by participant.'
            },
            categories: {
                type: jsPsych.plugins.parameterType.COMPLEX,
                default: [],
                pretty_name: 'Category labels',
                description: 'Labels of categories that participants can choose from.'
            },
            expectations: {
                type: jsPsych.plugins.parameterType.COMPLEX,
                default: [],
                pretty_name: 'Expected category labels',
                description: 'Labels of categories that participants are expected to choose.'
            },
            exclusive: {
                type: jsPsych.plugins.parameterType.BOOL,
                default: false,
                pretty_name: 'Exclusive labelling',
                description: 'Should each category be used only once?'
            },
            obligatory: {
                type: jsPsych.plugins.parameterType.BOOL,
                default: true,
                pretty_name: 'Obligatory answers',
                description: 'Should participants be required to categorise _all_ images?'
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
            categories: t.categories.join(),
            expectations: t.expectations.join(),
            exclusive: t.exclusive,
            obligatory: t.obligatory,
            rt: null
        };

        // extend data structure
        Object.keys(t.images).forEach(function(k, i) {
            data['image_' + i] = t.images[k];
            data['selection_' + i] = null;
            if (t.expectations.length == t.images.length) {
                data['expectation_' + i] = t.expectations[i];
                data['correct_' + i] = null;
            }
        });

        // setup trial container
        let trialContainer = $(
            '<fieldset style="border: none;">' + 
                '<div id="instruction-container" style="width: 1400px;" align="center">' + (!t.exclusive ? lang[t.study_language].instructions : lang[t.study_language].instructions_exclusive) + '</div>' + 
                '<div id="trial-container" style="width: 1400px; max-height:70vh; overflow-y:scroll; font-size: small; background: #eee;" align="center">' + 
                    '<div id="image-container" style="width: 75%;" align="justify">' + 
                        Object.entries(t.images).reduce((str, [p, val]) => {
                            return str + 
                                '<span style="">' + 
                                        '<img src="{0}" style="max-height: {1}px; height: auto;" />'.format(val, image_max_height) + 
                                        '<select name="labels-{0}" id="labels-{1}" data-ref="{2}">'.format(p.toString(), p.toString(), p.toString()) + 
                                            '<option value="" disabled selected hidden>{0}</option>'.format(lang[t.study_language].please_select) + 
                                            Object.entries(t.categories).reduce((str2, [p2, val2]) => {
                                                return str2 + 
                                                    '<option id="labels-{0}-opt-{1}" value="{2}">{3}</option>'.format(p.toString(), p2.toString(), val2, val2) + 
                                                    '\n';
                                            }, '') + 
                                        '</select>' + 
                                '</span>\n';
                        }, '') +
                    '</div>' + 
                '</div>' + 
                '<button id="reset-categories">' + lang[t.study_language].reset + '</button>' + 
                '&nbsp;&nbsp;&nbsp;' + 
                '<button id="submit-categories">' + lang[t.study_language].submit + '</button>' + 
            '</fieldset>'
        );

        // grab relevant elements
        let buttonReset = trialContainer.find('#reset-categories');
        let buttonSubmit = trialContainer.find('#submit-categories');
        let instructionContainer = trialContainer.find('#instruction-container');
        let imageContainer = trialContainer.find('#image-container');
        let selectAll = imageContainer.find('select');

        // add reset
        buttonReset.on('click', function(){
            $('#image-container select').each(function(i){
                $(this).val("");
            });
        });
        
        // add submission & checks
        buttonSubmit.on('click', function(){
            let currentSelection = [];
            let duplicate = false;
            let missing = false;

            $('#image-container select').each(function(i){
                if ((typeof(this.value) == 'undefined' || this.value == null || this.value == '') && t.obligatory) {
                    missing = true;
                    $(this).shake();
                    return;
                }

                if (this.value != '' && $.inArray(this.value, currentSelection) > -1 && t.exclusive) {
                    duplicate = true;
                    $(this).shake();
                    return;
                }
            
                currentSelection.push(this.value);
                data['selection_' + $(this).attr('data-ref')] = this.value;
                if (t.expectations.length == t.images.length) {
                    data['correct_' + $(this).attr('data-ref')] = data['expectation_' + $(this).attr('data-ref')] == this.value;
                }
            });

            if ((duplicate && t.exclusive) || (missing && t.obligatory)) {
                $(instructionContainer).shake();
                return;
            }

            data.rt = performance.now() - time_start;

            jsPsych.pluginAPI.clearAllTimeouts();
            jsPsych.finishTrial(data);
        });

        // add exclusive option criterion, if desired
        if (t.exclusive) {
            selectAll.on('change', function(){
                let currentSelection = [];
                let parent = this;

                $('#image-container select').each(function(i){
                    if (typeof(this.value) == 'undefined' || this.value == null || this.value == '' || this.id == parent.id) {
                        return;
                    }

                    currentSelection.push(this.value);
                });

                if ($.inArray(this.value, currentSelection) > -1) {
                    $('#image-container select').each(function(i){
                        if (typeof(this.value) == 'undefined' || this.value == null || this.value == '' || this.value != parent.value) {
                            return;
                        }

                        $(this).shake();
                    });

                    $(this).val("");
                }
            });
        }

        // clear, show & start measuring RT
        e.innerHTML = '';
        trialContainer.appendTo(e);
        const time_start = performance.now();
    }

    return plugin;
})();