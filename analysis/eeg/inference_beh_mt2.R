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
df_long <- "./data/preprocessed/beh/all_mt2.csv";


#### load and recode
data <- read.delim(df_long, header = TRUE, sep = ",", dec = ".");
data$sid <- factor(data$sid);
data$trial_no <- as.numeric(as.character(data$trial_no));
data$block <- as.numeric(as.character(data$block));
data$context <- factor(data$context);
data$face <- factor(data$face);
data$feature <- factor(data$feature);
data$stimulus <- factor(data$stimulus);
data$target <- factor(data$target);
data$alternative <- factor(data$alternative);
data$fit_t <- as.numeric(as.character(data$fit_t));
data$fit_a <- as.numeric(as.character(data$fit_a));
data$ufit_t <- as.numeric(as.character(data$ufit_t));
data$ufit_a <- as.numeric(as.character(data$ufit_a));
data$target_pos <- factor(data$target_pos);
data$is_bad <- as.logical(data$is_bad);
data$correct <- as.logical(data$correct);
data$rt <- as.numeric(as.character(data$rt));
data$log_rt <- log(data$rt);
data$z_no <- (data$trial_no - mean(data$trial_no)) / sd(data$trial_no);
data$z_fit <- (data$fit_t - mean(data$fit_t)) / sd(data$fit_t);
data$z_ufit <- (data$ufit_t - mean(data$ufit_t)) / sd(data$ufit_t);

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

model.m1 <- lmer(log_rt ~ z_no * is_bad * z_fit + (1+z_no:is_bad:z_fit|sid:context) + (1|sid:target_pos) + (1|sid/stimulus) + (1|feature:face:context), data = data.task);
# converges
isSingular(model.m1); # TRUE

model.m1 <- lmer(log_rt ~ z_no * is_bad * z_fit + (1+z_no:z_fit|sid:context) + (1|sid:target_pos) + (1|sid/stimulus) + (1|feature:face:context), data = data.task);
# converges
isSingular(model.m1); # TRUE

model.m1 <- lmer(log_rt ~ z_no * is_bad * z_fit + (1+z_no+z_fit|sid:context) + (1|sid:target_pos) + (1|sid/stimulus) + (1|feature:face:context), data = data.task);
# converges
isSingular(model.m1); # FALSE

# this is a bit of a nice surprise actually
# because this makes our model fitting
# procedure very quick. this is kind of
# everything we'd want in the model already.

### save best model & get summary
model.best <- model.m1;
results.sum <- summary(model.best);

### inspect model
qqnorm(resid(model.best))
qqline(resid(model.best))
# looking a-ok

### save results for figures
# save all responses
data.pred <- data.task;
data.pred$fitted <- fitted(model.best);
write.csv(data.pred, './data/processed/beh/mt2/fitted.csv');

# save model outputs
write.csv(results.sum$coefficients, './data/processed/beh/mt2/summary.csv');

# also do the emt real quick
data.trends <- test(emtrends(model.best, ~ is_bad * z_fit, var = "z_fit", at = list(z_fit = c(0)), pbkrtest.limit = 4097), pbkrtest.limit = 4097);
write.csv(data.trends, './data/processed/beh/mt2/trends.csv');
