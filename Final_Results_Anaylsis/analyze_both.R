library(tidyverse)

both_data <- read_csv("final_results/final_results_both.csv")
print(both_data)

both_plot_data <- group_by(both_data, Model) %>%
  summarise(mean_test_acc = mean(Test_Accuracy)) %>%
  print()

both_plot_data <- group_by(both_data, Model) %>%
  summarise(max_test_acc = max(Test_Accuracy)) %>%
  print()



mfcc_only <- filter(both_data, Dataset == "fulldata_normalized") %>%
  pivot_longer(Training_Accuracy:Test_Accuracy,
            names_to = "acc_type", values_to = "acc_score") %>%
  mutate(Model = factor(Model, levels=c("Perceptron", "MLP", "KNN", "SVM", "Decision_Tree", "Random_Forest", "Gradient_Boost"))) %>%
  #filter(acc_type == "Test_Accuracy") %>%
  print()

mfcc_plot = ggplot(data = mfcc_only, aes(x = Model, y = acc_score, fill = acc_type)) +
  geom_bar(stat = "identity", position = position_dodge(), alpha = 0.75)  +
  geom_hline(yintercept = 0.0416, linetype = "dashed", color = "red", size=1.5) + 
  ggtitle("Fulldata Normalized") + 
  scale_fill_manual(values = c("Training_Accuracy" = "#3e8bbe", "Test_Accuracy" = "#fabf40"),
                    name = "Test Accuracy",
                    labels = c("Training_Accuracy" = "Training Set", "Test_Accuracy" = "Test Set")) + 
  labs(x = "", y = "Test Set Accuracy Score", fill = "") + 
  theme_bw(15) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) 
  
ggsave("fulldata_normalized_plot.png", plot = mfcc_plot, width = 8, height = 4, units = "in")

show(mfcc_plot)


# raw_data <- read_csv("Datasets/mfcc_only.csv")
# print(unique(raw_data$species))
# print(length(unique(raw_data$species)))