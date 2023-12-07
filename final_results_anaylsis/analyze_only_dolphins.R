library(tidyverse)

dolphin_data <- read_csv("final_results/final_results_only_dolphins.csv")
print(dolphin_data)

dolphin_plot_data <- group_by(dolphin_data, Model) %>%
  summarise(mean_test_acc = mean(Test_Accuracy)) %>%
  print()

dolphin_plot_data <- group_by(dolphin_data, Model) %>%
  summarise(max_test_acc = max(Test_Accuracy)) %>%
  print()