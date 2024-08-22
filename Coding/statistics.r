library(ggplot2)
library(gridExtra)
library(ggsci)

data <- read.csv("data/classification_reports.csv")
data$data_type <- as.factor(data$folder)
data$total_frames <- as.factor(data$num_frames)
data$subset_frames <- as.factor(data$num_frames_reduced)
data$model <- as.factor(data$model)

cols = c("folder", "num_frames", "num_frames_reduced", "model")

data[cols] <- lapply(data[cols], as.factor)


# Statistical tests
model <- lm(log(accuracy) ~ model + data_type + total_frames + subset_frames , data = data)
par(mfrow = c(2,2))
summary(model)
anova(model)

aov_model <- aov(accuracy ~ model + data_type + total_frames + subset_frames, data = data)
tukey_result <- TukeyHSD(aov_model, "model")
print(tukey_result)



# Create boxplots
cbPalette <- c("#EAAA60", "#7DA6C6", "#84C3B7", "#E68B81", "#B7B2D0")

# Create four boxplots with increased axis label size
p1 <- ggplot(data, aes(x = folder, y = accuracy, fill = folder)) +
  geom_boxplot() + 
  scale_fill_manual(values = cbPalette) + 
  labs(title = "a.", x = "Data type", y = "Accuracy") +
  theme(axis.title = element_text(size = 14), # Increase axis label size
        axis.text = element_text(size = 12)) + 
  theme_classic() + theme(legend.position = "none")

p2 <- ggplot(data, aes(x = num_frames, y = accuracy, fill = num_frames)) +
  geom_boxplot() +
  scale_fill_manual(values = cbPalette) + 
  labs(title = "b.", x = "Total frames", y = "Accuracy") +
  theme(axis.title = element_text(size = 14),
        axis.text = element_text(size = 12)) + 
  theme_classic() + theme(legend.position = "none")

p3 <- ggplot(data, aes(x = num_frames_reduced, y = accuracy, fill = num_frames_reduced )) +
  geom_boxplot() +
  scale_fill_manual(values = cbPalette) + 
  labs(title = "c.", x = "Frames after subset", y = "Accuracy") +
  theme(axis.title = element_text(size = 14),
        axis.text = element_text(size = 12)) + 
  theme_classic() + theme(legend.position = "none")

p4 <- ggplot(data, aes(x = model, y = accuracy, fill = model)) +
  geom_boxplot() +
  scale_fill_manual(values = cbPalette) + 
  labs(title = "d.", x = "Model", y = "Accuracy") +
  theme(axis.title = element_text(size = 14),
        axis.text = element_text(size = 12)) + 
  theme_classic() + theme(legend.position = "none") +
  scale_x_discrete(labels = c("Random forest", "XGBoost"))

# Arrange the plots in a 2x2 grid
grid.arrange(p1, p2, p3, p4, nrow = 2, ncol = 2)

