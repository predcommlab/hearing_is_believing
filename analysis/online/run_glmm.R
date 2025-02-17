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
#setwd("C:\\Users\\fschneider\\Desktop\\code\\semantic_priors\\analyses\\exp2");
setwd("/users/fabianschneider/desktop/university/phd/fabian_semantic_priors.nosync/analyses/exp2/")
df_long <- "./data/behaviour_raw.csv";


#### load and recode
data <- read.delim(df_long, header = TRUE, sep = ",", dec = ".");
data$pid <- factor(data$pid);
data$no <- as.numeric(as.character(data$no));
data$time <- as.numeric(as.character(data$time));
data$block <- as.numeric(as.character(data$block));
data$kappa <- as.numeric(as.character(data$kappa));
data$fit_a <- as.numeric(as.character(data$fit_a));
data$fit_b <- as.numeric(as.character(data$fit_b));
data$rfit_a <- as.numeric(as.character(data$rfit_a));
data$rfit_b <- as.numeric(as.character(data$rfit_b));
data$repetition <- factor(data$repetition);
data$context_a <- factor(data$context_a);
data$context_b <- factor(data$context_b);
data$word_a <- factor(data$word_a);
data$word_b <- factor(data$word_b);
data$speaker_context_tgt <- factor(data$speaker_context_tgt);
data$speaker_context_alt <- factor(data$speaker_context_alt);
data$position_a <- factor(data$position_a);
data$speaker_face <- factor(data$speaker_face);
data$speaker_feature <- factor(data$speaker_feature);
data$strategy <- factor(data$strategy);
data$ismiss <- as.logical(data$ismiss);
data$isunreasonable <- as.logical(data$isunreasonable);
data$iscontrol <- as.logical(data$iscontrol);
data$chose_a <- as.logical(data$chose_a);
data$rt <- as.numeric(as.character(data$rt));
data$z_fit <- (data$fit_a - mean(data$fit_a)) / std(data$fit_a);
data$z_time <- (data$time - mean(data$time)) / std(data$time);
data$z_kappa <- (data$kappa - mean(data$kappa)) / std(data$kappa);

### control responses
data.control <- data[data$iscontrol == T,];
results.control.at <- aggregate(chose_a ~ pid, data = data.control, FUN = mean);
results.control.rt <- aggregate(rt ~ pid, data = data.control, FUN = function(x) { c(rt.min = min(x), rt.max = max(x), rt.mean = mean(x), rt.sd = sd(x), rt.med = median(x)); });
results.control.ms <- aggregate(isunreasonable ~ pid, data = data, FUN = function(ir) { sum(ir) }, na.action = NULL);
results.control.ms.rm <- c(ms.N = sum(results.control.ms$isunreasonable),
                           ms.mu = mean(results.control.ms$isunreasonable), 
                           ms.min = min(results.control.ms$isunreasonable), 
                           ms.max = max(results.control.ms$isunreasonable), 
                           ms.sd = sd(results.control.ms$isunreasonable), 
                           ms.med = median(results.control.ms$isunreasonable));
# a-ok, very nice

t.test(results.control.at$chose_a, mu = .5, alternative = "two.sided")
sd(results.control.at$chose_a)
# 	One Sample t-test
# 
# data:  results.control.at$chose_a
# t = 62.227, df = 34, p-value < 2.2e-16
# alternative hypothesis: true mean is not equal to 0.5
# 95 percent confidence interval:
#   0.9548807 0.9855955
# sample estimates:
#   mean of x 
# 0.9702381 
# sd [1] 0.044707

### make survey results available
results.survey.d <- data[!duplicated(data[,c('pid')]),c('pid', 'strategy')];
table(results.survey.d$strategy)


### get responses
data.task <- data[data$iscontrol == F & data$isunreasonable == F & data$ismiss == F,];


### start maximal modelling
# basic model
model.m1 <- glmer(chose_a ~ 
                    time*kappa + 
                    time*fit_a +
                    (1+time*fit_a+time*kappa|pid:context_a) + (1|pid:position_a), data = data.task, family = binomial(link = 'logit'),
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)));
# converges
isSingular(model.m1) # = TRUE
# but produces a singular fit


# try removing context_a specific slopes
model.m2 <- glmer(chose_a ~ 
                    time*kappa + 
                    time*fit_a +
                    (1+time*fit_a+time*kappa|pid) + (1|pid:context_a:context_b) + (1|pid:position_a), 
                  data = data.task, family = binomial(link = 'logit'),
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)));
# converges
isSingular(model.m2) # = TRUE


# try removing the position term (seems to not fit)
model.m3 <- glmer(chose_a ~ 
                    time*kappa + 
                    time*fit_a +
                    (1+time*fit_a+time*kappa|pid) + (1|pid:context_a:context_b), 
                  data = data.task, family = binomial(link = 'logit'),
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)));
# converges
isSingular(model.m3) # = TRUE


# try removing the intercept correlation (seems not to fit)
model.m4 <- glmer(chose_a ~ 
                    time*kappa + 
                    time*fit_a +
                    (0+fit_a+kappa|pid) + (1|pid:context_a:context_b), 
                  data = data.task, family = binomial(link = 'logit'),
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)));
# converges
isSingular(model.m4) # = TRUE


# try removing the time interaction (seems not to fit well)
model.m5 <- glmer(chose_a ~ 
                    time*kappa + 
                    time*fit_a +
                    (0+fit_a+kappa|pid) + (1|pid:context_a:context_b), 
                  data = data.task, family = binomial(link = 'logit'),
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)));
# converges
isSingular(model.m5) # = FALSE


# try adding the position term again
model.m6 <- glmer(chose_a ~ 
                    time*kappa + 
                    time*fit_a +
                    (0+fit_a+kappa|pid) + (1|pid:context_a:context_b) + (1|position_a), 
                  data = data.task, family = binomial(link = 'logit'),
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)));
# converges
isSingular(model.m6) # = TRUE


# try adding speaker terms
model.m7 <- glmer(chose_a ~ 
                    time*kappa + 
                    time*fit_a +
                    (0+fit_a+kappa|pid) + (1|pid:context_a:context_b) + (1|speaker_context_tgt:speaker_context_alt) + 
                    (1|pid:speaker_face:context_a) + (1|pid:speaker_feature:context_a), 
                  data = data.task, family = binomial(link = 'logit'),
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)));
# converges
isSingular(model.m7) # = TRUE


# try removing the original context terms because others might be better predictors
model.m8 <- glmer(chose_a ~ 
                    time*kappa + 
                    time*fit_a +
                    (0+fit_a+kappa|pid) + 
                    (1|speaker_face:context_a:context_b) + (1|pid:context_a:context_b), 
                  data = data.task, family = binomial(link = 'logit'),
                  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5)));
# converges
isSingular(model.m8) # = FALSE
# does this improve our predictions?
anova(model.m8, model.m5) # yes
#          npar    AIC    BIC  logLik deviance  Chisq Df Pr(>Chisq)    
# model.m5   10 8390.1 8460.4 -4185.0   8370.1                        
# model.m8   11 8235.1 8312.4 -4106.5   8213.1   157  1  < 2.2e-16 ***


### save best model & get summary
model.best <- model.m8;
results.sum <- summary(model.best);

### inspect model
sim <- simulateResiduals(fittedModel = model.best, plot = T)
testDispersion(sim) # looks ok
testZeroInflation(sim) # looks ok
plot(sim) # looks pretty good
plotResiduals(sim, form = data.task$time) # ok
plotResiduals(sim, form = data.task$kappa) # ok
plotResiduals(sim, form = data.task$fit_a) # ok
plotResiduals(sim, form = data.task$pid) # ok
# overall, I think we can be fairly happy
# with the model fit here.


### save results for figures
# save all responses
data.pred <- data.task;
data.pred$fitted <- fitted(model.best);
write.csv(data.pred, './results/glmm/fitted.csv');

# save model outputs
write.csv(results.sum$coefficients, './results/glmm/summary.csv');

# save control & survey data
write.csv(data.control, './results/glmm/control.csv');
write.csv(results.survey.d, './results/glmm/survey.csv');