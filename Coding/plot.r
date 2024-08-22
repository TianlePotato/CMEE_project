library(ggplot2)

# Read the dataset
data <- read.csv("data/best_sample_size.csv")
data <- data[-1]


# Fit the asymptotic regression model
model <- nls(accuracy ~ b0 + (b1 * sample_size) / (b2 + sample_size), 
             data = data, 
             start = list(b0 = min(data$accuracy), 
                          b1 = max(data$accuracy) - min(data$accuracy), 
                          b2 = median(data$sample_size)))

# Extract the plateau value (b0 + b1)
plateau_value <- coef(model)["b0"] + coef(model)["b1"]

# Predict accuracy for a range of sample sizes to plot the fitted curve
data$predicted_accuracy <- predict(model, newdata = data)

plot <- ggplot(data, aes(x = sample_size, y = accuracy)) + 
  geom_point(size = 4, alpha = 0.8) +
  xlab("Sample size") +
  ylab("Prediction accuracy") + 
  geom_line(aes(y = predicted_accuracy), color = "blue", linewidth = 1) + 
  geom_hline(yintercept = plateau_value, linetype = "dashed", color = "red") +
  annotate("text", x = max(data$sample_size), y = plateau_value, 
           label = paste("Plateau =", round(plateau_value, 3)), vjust = -1)

           


# Create the plot
plot <- ggplot(data, aes(x = sample_size, y = accuracy, shape = model, color = model)) + 
  xlab("Sample size") +
  ylab("Prediction accuracy") + 
  geom_point(size = 4, alpha = 0.8) +  # Increase size of points    
  theme_classic()

plot <- plot + theme(
    axis.text = element_text(size = 12),  # Increase size of axis ticks
    axis.title = element_text(size = 14),
    legend.text = element_text(size = 14),  # Increase size of legend text
    legend.title = element_text(size = 16)  # Increase size of legend title
  ) +
  guides(
    shape = guide_legend(override.aes = list(size = 6)),  # Increase size of shapes in legend
    color = guide_legend(override.aes = list(size = 6))   # Increase size of colors in legend
  ) + xlim(0, 1200) + ylim(0, 1)


# Save the plot
ggsave("Dataset/model_performance_plot.png", plot = plot, width = 8.69, height = 6.14, dpi = 300)
