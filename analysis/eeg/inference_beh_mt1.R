#### libraries
# required for linear mixed effects models
library(lme4);

# required for significance testing in LMMs
library(lmerTest);

# required for pretty plotting
library(ggplot2);

# required for colour-blind friendly palettes
library(viridis);

# required for estimating marginal means
library(emmeans);

# required for glmer diagnostics
library(DHARMa);


#### file management
# setup data files
setwd("/users/fabianschneider/desktop/university/phd/fabian_semantic_priors.nosync/analyses/exp4_eeg/")
df_long <- "./data/preprocessed/beh/all_mt1.csv";


#### load and recode
data <- read.delim(df_long, header = TRUE, sep = ",", dec = ".");
data$sid <- factor(data$sid);
data$trial_no <- as.numeric(as.character(data$trial_no));
data$block <- as.numeric(as.character(data$block));
data$context <- factor(data$context);
data$face <- factor(data$face);
data$feature <- factor(data$feature);
data$stimulus <- factor(data$stimulus);
data$word_a <- factor(data$word_a);
data$word_b <- factor(data$word_b);
data$kappa <- as.numeric(as.character(data$kappa));
data$fit_a <- as.numeric(as.character(data$fit_a));
data$fit_b <- as.numeric(as.character(data$fit_b));
data$a_pos <- factor(data$a_pos);
data$a_is_target <- as.logical(data$a_is_target);
data$chose_a <- as.logical(data$chose_a);
data$chose_context <- as.logical(data$chose_context);
data$rt <- as.numeric(as.character(data$rt));
data$log_rt <- log(data$rt);
data$z_no <- (data$trial_no - mean(data$trial_no)) / sd(data$trial_no);
data$z_fit <- (data$fit_a - mean(data$fit_a)) / sd(data$fit_a);

### real quick, check overall in-context performance
t.test(data$chose_context, alternative = 'two.sided');
# One Sample t-test
# 
# data:  data$chose_context
# t = 185.92, df = 8399, p-value < 2.2e-16
# alternative hypothesis: true mean is not equal to 0
# 95 percent confidence interval:
#   0.7960415 0.8130061
# sample estimates:
#   mean of x 
# 0.8045238 

### remove misses
data.task <- data[is.na(data$rt) == F,];
loss <- c(NROW(data) - NROW(data.task), 100 * (1 - NROW(data.task) / NROW(data)));

### start maximal modelling
# A quick explanation:
# Because we want to find the most generalisable model we can, we 
# are pursing a maximal modeling approach here. Effectively, that
# means that we keep our fixed effects constant, but try to capture
# the maximal amount of variance in random effects (i.e., that vary
# by participant or other random variables in our experiment) to 
# make sure that our effects of interest really are reliable.
# You can find more information about this approach here:
# 
#   Barr, D.J., Levy, R., Scheepers, C., Tily, H.J. (2013). Random effects structure for confirmatory hypothesis testing: Keep it maximal. Journal of Memory and Language, 68, 255-278. 10.1016/j.jml.2012.11.001
#   Yarkoni, T. (2020). The generalizability crisis. Behavioural and Brain Sciences, 45, e1. 10.1017/S0140525X20001685

# let's start with a nice and complex model
model.m1 <- glmer(chose_a ~ 
                    z_no*kappa + 
                    z_no*z_fit +
                    (1|sid:a_pos) + (1+z_no*z_fit+z_no*kappa|sid:context), data = data.task, family = binomial(link = 'logit'),
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)));
# converges
isSingular(model.m1); # TRUE

# it is a pretty complex RE structure, let's try something more simple first
model.m1 <- glmer(chose_a ~ 
                    z_no*kappa + 
                    z_no*z_fit +
                    (1|sid:a_pos) + (1+z_no+z_fit+kappa|sid:context), data = data.task, family = binomial(link = 'logit'),
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)));
# converges
isSingular(model.m1); # TRUE

# let's go even simpler then
model.m1 <- glmer(chose_a ~ 
                    z_no*kappa + 
                    z_no*z_fit +
                    (1+z_no+z_fit+kappa|sid:context), data = data.task, family = binomial(link = 'logit'),
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)));
# converges
isSingular(model.m1); # FALSE
# nice, baseline model

# let's try and incorporate interaction slopes again
model.m2 <- glmer(chose_a ~ 
                    z_no*kappa + 
                    z_no*z_fit +
                    (1+z_no*z_fit+z_no*kappa|sid:context), data = data.task, family = binomial(link = 'logit'),
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)));
# converges
isSingular(model.m2); # TRUE

# here's a different way that is also very reasonable
model.m2 <- glmer(chose_a ~ 
                    z_no*kappa + 
                    z_no*z_fit +
                    (1+z_no+z_fit|sid:context) + (0+kappa+z_no:kappa|sid), data = data.task, family = binomial(link = 'logit'),
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)));
# converges
isSingular(model.m2); # FALSE
# test it against previous
anova(model.m1, model.m2)
# Models:
#   model.m2: chose_a ~ z_no * kappa + z_no * z_fit + (1 + z_no + z_fit | sid:context) + 
#   model.m2:     (0 + kappa + z_no:kappa | sid)
# model.m1: chose_a ~ z_no * kappa + z_no * z_fit + (1 + z_no + z_fit + kappa | 
#                                                      model.m1:     sid:context)
# npar    AIC    BIC  logLik deviance Chisq Df Pr(>Chisq)    
# model.m2   15 8853.2 8958.3 -4411.6   8823.2                        
# model.m1   16 8816.7 8928.9 -4392.4   8784.7 38.42  1  5.704e-10 ***
 
# let's try adding in some more REs again to the baseline then
model.m2 <- glmer(chose_a ~ 
                    z_no*kappa + 
                    z_no*z_fit +
                    (1+z_no+z_fit+kappa|sid:context) + (1|sid:a_pos) + (1|face:feature), data = data.task, family = binomial(link = 'logit'),
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)));
# converges
isSingular(model.m2); # TRUE

# maybe remove face:feature
model.m2 <- glmer(chose_a ~ 
                    z_no*kappa + 
                    z_no*z_fit +
                    (1+z_no+z_fit+kappa|sid:context) + (1|sid:a_pos), data = data.task, family = binomial(link = 'logit'),
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)));
# converges
isSingular(model.m2); # FALSE
# okay let's test it
anova(model.m1, model.m2)
# Models:
# model.m1: chose_a ~ z_no * kappa + z_no * z_fit + (1 + z_no + z_fit + kappa | 
#                                                      model.m1:     sid:context)
# model.m2: chose_a ~ z_no * kappa + z_no * z_fit + (1 + z_no + z_fit + kappa | 
#                                                      model.m2:     sid:context) + (1 | face:feature)
# npar    AIC    BIC  logLik deviance  Chisq Df Pr(>Chisq)  
# model.m1   16 8816.7 8928.9 -4392.4   8784.7                       
# model.m2   17 8813.4 8932.6 -4389.7   8779.4 5.3304  1    0.02096 *
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# let's compare to a null model real quick
model.m0 <- glmer(chose_a ~ 
                    z_no*kappa + 
                    z_no*z_fit +
                    (1|sid:context), data = data.task, family = binomial(link = 'logit'),
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)));
# converges
isSingular(model.m0); # FALSE
# test against model
anova(model.m0, model.m2)
# Models:
#   model.m0: chose_a ~ z_no * kappa + z_no * z_fit + (1 | sid:context)
# model.m2: chose_a ~ z_no * kappa + z_no * z_fit + (1 + z_no + z_fit + kappa | 
#                                                      model.m2:     sid:context) + (1 | face:feature)
# npar    AIC    BIC  logLik deviance  Chisq Df Pr(>Chisq)    
# model.m0    7 9104.5 9153.6 -4545.3   9090.5                         
# model.m2   17 8813.4 8932.6 -4389.7   8779.4 311.12 10  < 2.2e-16 ***
#   ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# okay then, that was pretty quick.

### save best model & get summary
model.best <- model.m2;
results.sum <- summary(model.best);

### inspect model
sim <- simulateResiduals(fittedModel = model.best, plot = T)
testDispersion(sim) # looks ok
testZeroInflation(sim) # looks ok
plot(sim) # looks pretty good
plotResiduals(sim, form = data.task$z_no) # ok
plotResiduals(sim, form = data.task$kappa) # ok
plotResiduals(sim, form = data.task$z_fit) # ok
plotResiduals(sim, form = data.task$sid) # not ideal but acceptable
plotResiduals(sim, form = data.task$context) # not ideal but acceptable i think given residuals overall are good and this is a design thing
# overall, I think we can be fairly happy
# with the model fit here.

### save results for figures
# save all responses
data.pred <- data.task;
data.pred$fitted <- fitted(model.best);
write.csv(data.pred, './data/processed/beh/mt1/fitted.csv');

# save model outputs
write.csv(results.sum$coefficients, './data/processed/beh/mt1/summary.csv');
