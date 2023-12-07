library(tidyverse)

data <- read_csv("final_results/final_results_whale_vs_dolphins.csv")
print(data)

plot_data <- group_by(data, Model) %>%
  summarise(mean_test_acc = mean(Test_Accuracy)) %>%
  print()

plot_data <- group_by(data, Model) %>%
  summarise(max_test_acc = max(Test_Accuracy)) %>%
  print()