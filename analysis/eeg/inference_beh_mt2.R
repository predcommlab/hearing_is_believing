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
setwd("/users/fabianschneider/desktop/university/phd/hearing_is_believing/analysis/eeg/")
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
model.m1 <- lmer(log_rt ~ z_no * is_bad * z_fit + (1+z_no:is_bad:z_fit|sid:context) + (1|sid:target_pos) + (1|sid/stimulus) + (1|feature:face:context), data = data.task);
# converges
isSingular(model.m1); # TRUE

model.m1 <- lmer(log_rt ~ z_no * is_bad * z_fit + (1+z_no:z_fit|sid:context) + (1|sid:target_pos) + (1|sid/stimulus) + (1|feature:face:context), data = data.task);
# converges
isSingular(model.m1); # TRUE

model.m1 <- lmer(log_rt ~ z_no * is_bad * z_fit + (1+z_no+z_fit|sid:context) + (1|sid:target_pos) + (1|sid/stimulus) + (1|feature:face:context), data = data.task);
# fails to converge

model.m1 <- lmer(log_rt ~ z_no * is_bad * z_fit + (1+z_no+z_fit|sid:context) + (1|sid:target_pos) + (1|sid/stimulus), data = data.task);
# converges
isSingular(model.m1); # FALSE

# try readding the slope for bads
model.m2 <- lmer(log_rt ~ z_no * is_bad * z_fit + (1+z_no+is_bad+z_fit|sid:context) + (1|sid:target_pos) + (1|sid/stimulus), data = data.task);
# converges
isSingular(model.m2); # TRUE

# try adding visual features
model.m2 <- lmer(log_rt ~ z_no * is_bad * z_fit + (1+z_no+z_fit|sid:context) + (1|sid:target_pos) + (1|sid/stimulus) + (1|feature:face), data = data.task);
# converges
isSingular(model.m2); # TRUE

# try adding individually
model.m2 <- lmer(log_rt ~ z_no * is_bad * z_fit + (1+z_no+z_fit|sid:context) + (1|sid:target_pos) + (1|sid/stimulus) + (1|feature) + (1|face), data = data.task);
# converges
isSingular(model.m2); # TRUE

# try only the face
model.m2 <- lmer(log_rt ~ z_no * is_bad * z_fit + (1+z_no+z_fit|sid:context) + (1|sid:target_pos) + (1|sid/stimulus) + (1|face), data = data.task);
# converges
isSingular(model.m2); # TRUE

# only feature instead?
model.m2 <- lmer(log_rt ~ z_no * is_bad * z_fit + (1+z_no+z_fit|sid:context) + (1|sid:target_pos) + (1|sid/stimulus) + (1|feature), data = data.task);
# converges
isSingular(model.m2); # TRUE

# well, this was quick.

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
data.trends <- test(emtrends(model.best, ~ is_bad * z_fit, var = "z_fit"));
write.csv(data.trends, './data/processed/beh/mt2/trends.csv');