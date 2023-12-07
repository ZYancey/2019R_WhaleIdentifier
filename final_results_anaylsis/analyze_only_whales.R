library(tidyverse)

whale_data <- read_csv("final_results/final_results_only_whales.csv")
print(whale_data)

whale_plot_data <- group_by(whale_data, Model) %>%
  summarise(mean_test_acc = mean(Test_Accuracy)) %>%
  print()

whale_plot_data <- group_by(whale_data, Model) %>%
  summarise(max_test_acc = max(Test_Accuracy)) %>%
  print()