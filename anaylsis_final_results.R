library(tidyverse)

whale_data = read_csv('final_results_only_whales.csv')
dolphine_and_whale_data = read_csv('final_results_with_dolphines.csv')
print(whale_data)
print(dolphine_and_whale_data)

# show the mean test accuracy
print("Dolphine mean test accuracy")
dolphine_plot_data = group_by(dolphine_and_whale_data, Model) %>%
    summarize(mean_test_acc = mean(Test_Accuracy)) %>%
    print()

# show the max test accuracy
print("Dolphine max test accuracy")
dolphine_plot_data = group_by(dolphine_and_whale_data, Model) %>%
    summarize(max_test_acc = max(Test_Accuracy)) %>%
    print()

# show the mean test accuracy
print("Whale mean test accuracy")
whale_plot_data = group_by(whale_data, Model) %>%
    summarize(mean_test_acc = mean(Test_Accuracy)) %>%
    print()

whale_plot = ggplot(whale_plot_data, aes(x=Model, mean_test_acc, fill=Model)) + 
    geom_col() +
    theme_bw()

#show(whale_plot)

# show the max test accuracy score per model
print("Whale max test accuracy")
whale_plot_data = group_by(whale_data, Model) %>%
    summarize(max_test_acc = max(Test_Accuracy)) %>%
    print()

whale_plot = ggplot(whale_plot_data, aes(x=Model, max_test_acc, fill=Model)) + 
    geom_col() +
    theme_bw()

#show(whale_plot)

# show the max test accuracy score per dataset type
whale_plot_data = group_by(whale_data, Dataset) %>%
    summarize(max_test_acc = max(Test_Accuracy))

whale_plot = ggplot(whale_plot_data, aes(x=Dataset, max_test_acc, fill=Dataset)) + 
    geom_col() +
    theme_bw()

show(whale_plot)