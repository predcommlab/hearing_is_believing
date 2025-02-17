#### libraries
# required for linear mixed effects models
library(lme4);

# required for significance testing in LMMs
library(lmerTest);

# required for pretty plotting
library(ggplot2);

# required for colour-blind friendly palettes
library(viridis);


#### file management
# setup data files
setwd("/users/fabianschneider/Desktop/university/PhD/hearing_is_believing/analysis/validation/");
df_long <- "./data/data.csv";
df_part <- "./data/prolific_pids.csv";


#### descriptives of sample
desc.pids <- read.delim(df_part, header = TRUE, sep = ",", dec = ".");
desc.age <- c(mean(desc.pids$Age), median(desc.pids$Age), min(desc.pids$Age), max(desc.pids$Age));
desc.sex <- c(NROW(desc.pids[desc.pids$Sex == "Female",]), NROW(desc.pids[desc.pids$Sex == "Male",]), NROW(desc.pids[desc.pids$Sex != "Female" & desc.pids$Sex != "Male",]))
desc.time <-c(mean(desc.pids$Time.taken), median(desc.pids$Time.taken), min(desc.pids$Time.taken), max(desc.pids$Time.taken));


#### load and code data
data <- read.delim(df_long, header = TRUE, sep = ",", dec = ".");
data$pid <- factor(data$pid);
data$no <- as.numeric(as.character(data$no));
data$is_control <- as.logical(data$is_control);
data$rt <- as.numeric(as.character(data$rt));
data$choice_is_target <- as.integer(as.logical(data$choice_is_target));
data$choice_is_popular <- as.integer(!as.logical(data$choice_is_target));
data$block <- factor(data$block);
data$r <- factor(data$r);
data$target_position <- factor(data$target_position);
data$target <- factor(data$target);
data$popular <- factor(data$popular);
data$context <- factor(data$context);
data$F_t <- as.numeric(as.character(data$F_t));
data$F_d <- as.numeric(as.character(data$F_d));
data$tdcs <- as.numeric(as.character(data$tdcs));
data$ad <- as.numeric(as.character(data$ad));
data$F_r <- log10(data$F_d) / log10(data$F_t);


#### control response bias
data.control <- data[data$is_control == TRUE,];
results.control.rb <- aggregate(choice_is_target ~ pid, data = data.control, FUN = mean);
# a-ok! everybody scored way over 60%


#### control reaction times
data.sub <- data[data$is_control == FALSE & data$no > -1,];
results.control.rt <- aggregate(rt ~ pid, data = data.sub, FUN = min);
# a-ok! RT boundaries mirror our pilot data


#### control misses
results.control.ms <- aggregate(rt ~ pid, data = data.sub, FUN = function(rt){ sum(is.na(rt)) }, na.action = NULL);
# a-ok! everybody way below 10%


#### get clean data, add coded values
data.clean <- data.sub[!is.na(data.sub$rt),];
data.clean$z_trial_no <- (data.clean$no - mean(data.clean$no)) / sd(data.clean$no);
data.clean$z_F_r <- (data.clean$F_r - mean(data.clean$F_r)) / sd(data.clean$F_r);
F.pca <- prcomp(t(rbind(log10(data.clean$F_t), log10(data.clean$F_d))), center = TRUE, scale = TRUE, rank = 1, retx = TRUE);
data.clean$F_m <- as.numeric(as.character(F.pca$x));


#### stats by target
results.items <- aggregate(choice_is_popular ~ target, data = data.clean, FUN = mean);
results.items.r <- aggregate(choice_is_popular ~ r:target, data = data.clean, FUN = mean);
results.items.c <- aggregate(choice_is_popular ~ context, data = data.clean, FUN = mean);
results.items.n <- aggregate(choice_is_popular ~ r, data = data.clean, FUN = mean);
thr = 0.15;
results.item.thr <- aggregate(choice_is_popular ~ context:target, data = data.clean, FUN = mean);
results.item.thr <- aggregate(choice_is_popular ~ context, data = results.item.thr, FUN = function(p){ sum(p >= thr & p <= 1-thr) });

#### item-level descriptives
results.items$popular <- "None";
results.items$context <- "None";
results.items$k <- (results.items$choice_is_popular - 0.5) / (1 - 0.5);
results.items$k1 <- 0;
results.items$k2 <- 0;
results.items$tdcs <- 0;
results.items$ad1 <- 0;
results.items$ad2 <- 0;
results.items$z_F_r <- 0;
results.items$F_m <- 0;
results.items$X <- 0;
results.items$p <- 0;
results.items$file1 <- "None";
results.items$file2 <- "None";

for (target in results.items$target) {
  results.items[results.items$target == target,]$popular <- as.character(data.clean[data.clean$target == target,][1,]$popular);
  results.items[results.items$target == target,]$tdcs <- data.clean[data.clean$target == target,][1,]$tdcs;
  results.items[results.items$target == target,]$ad1 <- data.clean[data.clean$target == target & data.clean$r == 1,][1,]$ad;
  results.items[results.items$target == target,]$ad2 <- data.clean[data.clean$target == target & data.clean$r == 2,][1,]$ad;
  results.items[results.items$target == target,]$z_F_r <- data.clean[data.clean$target == target,][1,]$z_F_r;
  results.items[results.items$target == target,]$F_m <- data.clean[data.clean$target == target,][1,]$F_m;
  X <- chisq.test(c(sum(data.clean$target == target & data.clean$choice_is_popular), sum(data.clean$target == target & data.clean$choice_is_target)));
  results.items[results.items$target == target,]$X <- X$statistic;
  results.items[results.items$target == target,]$p <- X$p.value;
  results.items[results.items$target == target,]$context <- as.character(data.clean[data.clean$target == target,][1,]$context);
  results.items[results.items$target == target,]$file1 <- as.character(data.clean[data.clean$target == target & data.clean$r == 1,][1,]$stimulus);
  results.items[results.items$target == target,]$file2 <- as.character(data.clean[data.clean$target == target & data.clean$r == 2,][1,]$stimulus);
  results.items[results.items$target == target,]$k1 <- results.items.r[results.items.r$target == target & results.items.r$r == 1,][1,]$choice_is_popular;
  results.items[results.items$target == target,]$k2 <- results.items.r[results.items.r$target == target & results.items.r$r == 2,][1,]$choice_is_popular;
}

results.items$k1 <- (results.items$k1 - 0.5) / (1 - 0.5);
results.items$k2 <- (results.items$k2 - 0.5) / (1 - 0.5);
results.items$context <- factor(results.items$context);
results.items$thr <- results.items$choice_is_popular >= .125 & results.items$choice_is_popular <= .875;
results.items.sum <- aggregate(thr ~ context, data = results.items, FUN = sum);
write.csv(results.items, "./results/items.csv", row.names = FALSE);
write.csv(results.items.sum, "./results/contexts.csv", row.names = FALSE);


#### model responses
mmm <- glmer(choice_is_popular ~ z_trial_no*ad + z_trial_no*tdcs + z_F_r + F_m + r + (1+z_trial_no | pid : target_position),
             data = data.clean,
             family = binomial(link = 'logit'));

lmm <- glmer(choice_is_popular ~ z_trial_no*ad + z_trial_no*tdcs + poly(z_F_r*F_m, 3) + (1 | pid : target_position),
             data = data.clean,
             family = binomial(link = 'logit'));

lme <- lmer(log10(rt) ~ z_trial_no*ad + z_trial_no*tdcs + ad*F_m*F_r + (1+z_trial_no | pid : target_position), data = data.clean)
