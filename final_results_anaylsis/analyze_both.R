library(tidyverse)

both_data <- read_csv("final_results/final_results_both.csv")
print(both_data)

both_plot_data <- group_by(both_data, Model) %>%
  summarise(mean_test_acc = mean(Test_Accuracy)) %>%
  print()

both_plot_data <- group_by(both_data, Model) %>%
  summarise(max_test_acc = max(Test_Accuracy)) %>%
  print()


mfcc_only <- filter(both_data, Dataset == "mfcc_only") %>%
  pivot_longer(Training_Accuracy:Test_Accuracy,
            names_to = "acc_type", values_to = "acc_score") %>%
  print()

mfcc_plot = ggplot(data = mfcc_only, aes(x = Model, y = acc_score, fill = acc_type)) +
  geom_bar(stat = "identity", position = position_dodge(), alpha = 0.75)  +
  theme_bw(20)

show(mfcc_plot)