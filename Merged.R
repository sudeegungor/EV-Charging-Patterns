library(dplyr)
library(ggplot2)
library(psych)
library(randomForest)
library(caret)
library(doParallel)
library(reshape2)
library(gridExtra)
library(caret)
library(fastDummies)
library(infotheo)
library(class)
library(e1071)
library(neuralnet)
library(NeuralNetTools)


# Read the dataset
#vehicleDataDF <- read.table("ev_charging_patterns.csv", header = TRUE, sep = ",", dec = ".")

# Function to correct illogical values in the data set
clean_ev_data <- function(input_file, output_file) {
  # Load the CSV file
  df <- read.csv(input_file)
  
  # Clean the data
  df <- df %>%
    # 1. Adjust energy consumption for DC Fast Charger
    mutate(`Energy.Consumed..kWh.` = ifelse(`Charger.Type` == 'DC Fast Charger' & `Energy.Consumed..kWh.` < 5, 5, `Energy.Consumed..kWh.`)) %>%
    
    # 2. Correct charging duration for extreme values
    mutate(`Charging.Duration..hours.` = ifelse(`Charging.Duration..hours.` < 0.1 | `Charging.Duration..hours.` > 12, 1.0, `Charging.Duration..hours.`)) %>%
    
    # 3. Fix logical inconsistencies in State of Charge (End %) and State of Charge (Start %)
    mutate(`State.of.Charge..End...` = ifelse(`State.of.Charge..End...` < `State.of.Charge..Start...`, `State.of.Charge..Start...`, `State.of.Charge..End...`)) %>%
    
    # 4. Handle invalid combinations of vehicle age and battery capacity
    mutate(`Battery.Capacity..kWh.` = ifelse(`Vehicle.Age..years.` < 1 & `Battery.Capacity..kWh.` < 5, 10, `Battery.Capacity..kWh.`)) %>%
    
    # 5. Adjust temperature and distance for logical inconsistencies
    mutate(`Temperature..C.` = ifelse(`Temperature..C.` > 40 & `Distance.Driven..since.last.charge...km.` > 200, 25, `Temperature..C.`)) %>%
    
    # 6. Fix extreme relationships between energy consumed and charging duration
    mutate(`Energy.Consumed..kWh.` = ifelse(`Energy.Consumed..kWh.` / `Charging.Duration..hours.` > 50, `Charging.Duration..hours.` * 50, `Energy.Consumed..kWh.`)) %>%
    
    # 7. Assign random values for Long-Distance Traveler
    mutate(`Distance.Driven..since.last.charge...km.` = ifelse(`User.Type` == 'Long-Distance Traveler', runif(sum(`User.Type` == 'Long-Distance Traveler'), 350, 1000), `Distance.Driven..since.last.charge...km.`),
           `Charging.Duration..hours.` = ifelse(`User.Type` == 'Long-Distance Traveler', runif(sum(`User.Type` == 'Long-Distance Traveler'), 3, 10), `Charging.Duration..hours.`)) %>%
    
    # 8. Assign random values for Casual Driver
    mutate(`Distance.Driven..since.last.charge...km.` = ifelse(`User.Type` == 'Casual Driver', runif(sum(`User.Type` == 'Casual Driver'), 50, 350), `Distance.Driven..since.last.charge...km.`),
           `Charging.Duration..hours.` = ifelse(`User.Type` == 'Casual Driver', runif(sum(`User.Type` == 'Casual Driver'), 1, 3), `Charging.Duration..hours.`)) %>%
    
    # 9. Calculate energy consumption and cost based on charging duration
    mutate(`Energy.Consumed..kWh.` = `Charging.Duration..hours.` * 10,
           `Charging.Cost..USD.` = `Energy.Consumed..kWh.` * 0.15)
  
  # Save the cleaned dataset
  write.csv(df, output_file, row.names = FALSE)
  
  # Return the cleaned dataset
  return(df)
}

input_file <- "ev_charging_patterns.csv"# raw data set
output_file <- "cleaned_ev_charging_patterns.csv"# after converting to logical numbers
vehicleDataDF <- clean_ev_data(input_file, output_file)



# Drop the features, because User.ID and Charging.Station.ID are useless besides Charging.Start.Time, Charging.End.Time have 1320 different values(these don't have any effect)
vehicleDataDF <- vehicleDataDF %>%
  select(-`User.ID`, -`Charging.Start.Time`, -`Charging.End.Time`, 
         -`Charging.Station.ID`, -`Vehicle.Age..years.`)

# Identify continuous and categorical features
continuous_features <- names(vehicleDataDF)[sapply(vehicleDataDF, is.numeric)]
categorical_features <- names(vehicleDataDF)[sapply(vehicleDataDF, is.factor) | sapply(vehicleDataDF, is.character)]
print(categorical_features)
print(continuous_features)
# Function for continuous features
continuous_report <- function(data, column) {
  stats <- data %>%
    summarise(
      Count = n(),
      Missing = sum(is.na(.data[[column]])),
      Cardinality = length(unique(.data[[column]])),
      Min = min(.data[[column]], na.rm = TRUE),
      Q1 = quantile(.data[[column]], 0.25, na.rm = TRUE),
      Mean = mean(.data[[column]], na.rm = TRUE),
      Median = median(.data[[column]], na.rm = TRUE),
      Q3 = quantile(.data[[column]], 0.75, na.rm = TRUE),
      Max = max(.data[[column]], na.rm = TRUE),
      SD = sd(.data[[column]], na.rm = TRUE)
    )
  stats <- cbind(Feature = column, stats)
  return(stats)
}

# Function for categorical features with additional statistics
categorical_report <- function(data, column) {
  freq_table <- table(data[[column]])
  freq_sorted <- sort(freq_table, decreasing = TRUE)
  
  # Calculate mode percent
  mode_percent <- freq_sorted[1] / length(data[[column]]) * 100
  
  # Calculate second mode percent (if available)
  second_mode_freq <- ifelse(length(freq_sorted) > 1, freq_sorted[2], NA)
  second_mode_percent <- ifelse(!is.na(second_mode_freq), second_mode_freq / length(data[[column]]) * 100, NA)
  
  # Create the summary statistics
  stats <- data %>%
    summarise(
      Count = n(),
      Missing = sum(is.na(.data[[column]])),
      Cardinality = length(unique(.data[[column]])),
      Mode = names(freq_sorted)[1],
      ModeFreq = freq_sorted[1],
      ModePercent = mode_percent,
      SecondMode = ifelse(length(freq_sorted) > 1, names(freq_sorted)[2], NA),
      SecondModeFreq = second_mode_freq,
      SecondModePercent = second_mode_percent
    )
  
  stats <- cbind(Feature = column, stats)
  return(stats)
}

# Function to generate bar plots for categorical features
generate_bar_plot <- function(data, column, title, x_label, y_label) {
  ggplot(data, aes_string(x = column)) + 
    geom_bar(fill = "steelblue") + 
    labs(title = title, x = x_label, y = y_label)
}

# Function to generate histograms for continuous features
generate_histogram <- function(data, column, title, x_label, y_label) {
  ggplot(data, aes_string(x = column)) + 
    geom_histogram(binwidth = 1, fill = "steelblue", color = "black") + 
    labs(title = title, x = x_label, y = y_label) + 
    theme_minimal()
}
# Imputation function

impute_na <- function(data) {
  # Impute continuous features with the mean
  for (col in continuous_features) {
    data[[col]][is.na(data[[col]])] <- mean(data[[col]], na.rm = TRUE)
  }
  
  # Impute categorical features with the mode
  for (col in categorical_features) {
    mode_value <- names(sort(table(data[[col]]), decreasing = TRUE))[1]
    data[[col]][is.na(data[[col]])] <- mode_value
  }
  
  return(data)
}

# Clamp transformation for handling outliers
clamp_outliers <- function(data, column) {
  Q1 <- quantile(data[[column]], 0.25, na.rm = TRUE)
  Q3 <- quantile(data[[column]], 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  
  lower_threshold <- Q1 - 1.5 * IQR
  upper_threshold <- Q3 + 1.5 * IQR
  
  # Apply clamping
  data[[column]] <- ifelse(
    data[[column]] < lower_threshold, lower_threshold,
    ifelse(data[[column]] > upper_threshold, upper_threshold, data[[column]])
  )
  
  return(data)
}



#-------------------------------------------------------------------------

# Print the first few rows and column names
print(dim(vehicleDataDF))
for (feature in continuous_features) {
  print(
    ggplot(vehicleDataDF, aes_string(x = feature, fill = "User.Type")) +
      geom_histogram(position = "dodge", bins = 30, alpha = 0.7) +
      labs(title = paste("Histogram of", feature, "by User.Type"),
           x = feature, y = "Frequency") +
      theme_minimal() +
      scale_fill_brewer(palette = "Set1", name = "User.Type")
  )
}
for (feature in categorical_features) {
  print(
    ggplot(vehicleDataDF, aes_string(x = feature, fill = "User.Type")) +
      geom_bar(position = "dodge", alpha = 0.7) +
      labs(title = paste("Bar Chart of", feature, "by User.Type"),
           x = feature, y = "Count") +
      theme_minimal() +
      scale_fill_brewer(palette = "Set1", name = "User.Type") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  )
}
print(colnames(vehicleDataDF))
str(vehicleDataDF)
head(vehicleDataDF)
# Replace empty strings with NA
vehicleDataDF <- vehicleDataDF %>%
  mutate(across(everything(), ~ ifelse(trimws(.) == "", NA, .)))

# Generate plots for categorical features
lapply(categorical_features, function(col) {
  print(generate_bar_plot(vehicleDataDF, col, 
                          paste("Distribution of", col), col, "Count"))
})

# Generate histograms for continuous features
lapply(continuous_features, function(col) {
  print(generate_histogram(vehicleDataDF, col, 
                           paste("Histogram of", col), col, "Frequency"))
  
})

# Encoding categorical variables as factors
vehicleDataDF[categorical_features] <- lapply(vehicleDataDF[categorical_features], as.factor)

# Descriptive statistics for the dataset
describe(vehicleDataDF)

# Generate reports dynamically for continuous features
continuous_report_df <- do.call(rbind, lapply(continuous_features, function(col) continuous_report(vehicleDataDF, col)))
# 
# # Generate reports dynamically for categorical features
categorical_report_df <- do.call(rbind, lapply(categorical_features, function(col) categorical_report(vehicleDataDF, col)))

# Combine reports into a single list for clarity
data_quality_report <- list(
  Continuous = continuous_report_df,
  Categorical = categorical_report_df
)
# 
# Print the reports
cat("Continuous Features Data Quality Report:\n")
print(data_quality_report$Continuous)

cat("\nCategorical Features Data Quality Report:\n")
#print(data_quality_report$Categorical)

#imputation
vehicleDataDF <- impute_na(vehicleDataDF)

#continuous_report_df_af <- do.call(rbind, lapply(continuous_features, function(col) continuous_report(vehicleDataDF, col)))
#data_quality_report <- list(
# Continuous_af = continuous_report_df_af)

#cat("Continuous Features Data Quality Report after imputation:\n")
#print(data_quality_report$Continuous_af)
# Apply clamping to all continuous features
for (col in continuous_features) {
  vehicleDataDF <- clamp_outliers(vehicleDataDF, col)
}


# Z-score normalization
z_score_normalization <- function(data) {
  data %>%
    mutate(
      Battery.Capacity..kWh. = scale(Battery.Capacity..kWh.),
      Energy.Consumed..kWh. = scale(Energy.Consumed..kWh.),
      Charging.Duration..hours. = scale(Charging.Duration..hours.),
      Charging.Rate..kW. = scale(Charging.Rate..kW.),
      Charging.Cost..USD. = scale(Charging.Cost..USD.),
      Distance.Driven..since.last.charge...km. = scale(Distance.Driven..since.last.charge...km.),
      Temperature..C. = scale(Temperature..C.),
      State.of.Charge..Start... = scale(State.of.Charge..Start...),
      State.of.Charge..End... = scale(State.of.Charge..End...)
    )
}


# Min-Max normalization
min_max_normalization <- function(data) {
  data %>%
    mutate(
      Battery.Capacity..kWh. = (Battery.Capacity..kWh. - min(Battery.Capacity..kWh., na.rm = TRUE)) / (max(Battery.Capacity..kWh., na.rm = TRUE) - min(Battery.Capacity..kWh., na.rm = TRUE)),
      Energy.Consumed..kWh. = (Energy.Consumed..kWh. - min(Energy.Consumed..kWh., na.rm = TRUE)) / (max(Energy.Consumed..kWh., na.rm = TRUE) - min(Energy.Consumed..kWh., na.rm = TRUE)),
      Charging.Duration..hours. = (Charging.Duration..hours. - min(Charging.Duration..hours., na.rm = TRUE)) / (max(Charging.Duration..hours., na.rm = TRUE) - min(Charging.Duration..hours., na.rm = TRUE)),
      Charging.Rate..kW. = (Charging.Rate..kW. - min(Charging.Rate..kW., na.rm = TRUE)) / (max(Charging.Rate..kW., na.rm = TRUE) - min(Charging.Rate..kW., na.rm = TRUE)),
      Charging.Cost..USD. = (Charging.Cost..USD. - min(Charging.Cost..USD., na.rm = TRUE)) / (max(Charging.Cost..USD., na.rm = TRUE) - min(Charging.Cost..USD., na.rm = TRUE)),
      Distance.Driven..since.last.charge...km. = (Distance.Driven..since.last.charge...km. - min(Distance.Driven..since.last.charge...km., na.rm = TRUE)) / (max(Distance.Driven..since.last.charge...km., na.rm = TRUE) - min(Distance.Driven..since.last.charge...km., na.rm = TRUE)),
      Temperature..C. = (Temperature..C. - min(Temperature..C., na.rm = TRUE)) / (max(Temperature..C., na.rm = TRUE) - min(Temperature..C., na.rm = TRUE)),
      State.of.Charge..Start... = (State.of.Charge..Start... - min(State.of.Charge..Start..., na.rm = TRUE)) / (max(State.of.Charge..Start..., na.rm = TRUE) - min(State.of.Charge..Start..., na.rm = TRUE)),
      State.of.Charge..End... = (State.of.Charge..End... - min(State.of.Charge..End..., na.rm = TRUE)) / (max(State.of.Charge..End..., na.rm = TRUE) - min(State.of.Charge..End..., na.rm = TRUE))
    )
}


vehicleDataDF_preprocessedSVM <- z_score_normalization(vehicleDataDF)
vehicleDataDF_preprocessedSVM <- min_max_normalization(vehicleDataDF)

#--------------------------------------------------------------------------------



# Convert 'User.Type' to factor if it is not already
vehicleDataDF$`User.Type` <- as.factor(vehicleDataDF$`User.Type`)

# Set the categorical features (if you have character features, they will be included automatically)
categorical_features <- names(vehicleDataDF)[sapply(vehicleDataDF, is.factor) | sapply(vehicleDataDF, is.character)]

#---------------------------------END OF PREPROCESSING------------------------------------

#---------------------------------SPLIT TRAIN AND TEST - START-----------------------------------

vehicleDataDF$User.Type <- as.factor(vehicleDataDF$User.Type)
vehicleDataDF$Vehicle.Model <- as.factor(vehicleDataDF$Vehicle.Model)
vehicleDataDF$Charging.Station.Location <- as.factor(vehicleDataDF$Charging.Station.Location)
vehicleDataDF$Charger.Type <- as.factor(vehicleDataDF$Charger.Type)
vehicleDataDF$Day.of.Week <- as.factor(vehicleDataDF$Day.of.Week)
vehicleDataDF$Time.of.Day <- as.factor(vehicleDataDF$Time.of.Day)
#colnames(vehicleDataDF_encoded)

# Shuffle the dataset
#set.seed(424242)  # For reproducibility
#shuffled_df <- vehicleDataDF[sample(nrow(vehicleDataDF)), ]

# Split the data into training (80%) and testing (20%) sets
#train_index <- sample(1:nrow(shuffled_df), size = 0.8 * nrow(shuffled_df))
#train_data <- shuffled_df[train_index, ]
#test_data <- shuffled_df[-train_index, ]
# Split data into train and test sets
set.seed(2443)
samp <- floor(0.8 * nrow(vehicleDataDF))
train_index <- sample(seq_len(nrow(vehicleDataDF)), size = samp)
train_data <- vehicleDataDF[train_index, ]
test_data <- vehicleDataDF[-train_index, ]

#---------------------------------SPLIT TRAIN AND TEST - END-----------------------------------


#---------------------------------NEURAL NETWORK MODEL------------------------------------

#---------------------------------nn_model_with_hyperparameter using neuralnet library START-----------------------------------
# Neural Network Evaluation with Multiple Hidden Layers and Visualization
colnames(vehicleDataDF_encoded)
nn_model_with_hyperparameter <- function(vehicleDataDF_encoded, hidden_layers_list, threshold = 0.01, seed = 4232,train_index,train_data,test_data) {
  # Encode categorical features
  colnames(vehicleDataDF_encoded)
  vehicleDataDF_encoded <- fastDummies::dummy_cols(
    vehicleDataDF,
    select_columns = c("Vehicle.Model", "Charging.Station.Location", "Time.of.Day", "Day.of.Week", "Charger.Type", "User.Type"),
    remove_first_dummy = FALSE,
    remove_selected_columns = TRUE
  )
  colnames(vehicleDataDF_encoded)
  
  # Normalize columns
  colnames(vehicleDataDF_encoded) <- gsub(" ", "_", colnames(vehicleDataDF_encoded))
  colnames(vehicleDataDF_encoded) <- gsub("-", "_", colnames(vehicleDataDF_encoded))
  vehicleDataDF_encoded <- min_max_normalization(vehicleDataDF_encoded)
  colnames(vehicleDataDF_encoded)
  
  # Split data into training and test sets
  set.seed(seed)
  samp <- floor(0.8 * nrow(vehicleDataDF_encoded))
  train_index <- sample(seq_len(nrow(vehicleDataDF_encoded)), size = samp)
  train_data <- vehicleDataDF_encoded[train_index, ]
  test_data <- vehicleDataDF_encoded[-train_index, ]
  
  input_features <- setdiff(colnames(vehicleDataDF_encoded), grep("User.Type_", colnames(vehicleDataDF_encoded), value = TRUE))
  output_features <- grep("User.Type_", colnames(vehicleDataDF_encoded), value = TRUE)
  formula <- as.formula(paste(paste(output_features, collapse = "+"), "~", paste(input_features, collapse = "+")))
  
  # Grid search results storage
  grid_search_results <- data.frame()
  
  for (hidden_layers in hidden_layers_list) {
    # Train Neural Network
    nn_model <- neuralnet(
      formula,
      data = train_data,
      hidden = hidden_layers,
      threshold = threshold,
      linear.output = FALSE,
      stepmax = 1e6,
      lifesign = "minimal"
    )
    
    # Predictions on test data
    pr.nn_test <- compute(nn_model, test_data[, input_features])
    pred_test <- apply(pr.nn_test$net.result, 1, which.max)
    true_test <- apply(test_data[, output_features], 1, which.max)
    
    # Metrics Calculation
    conf_matrix <- confusionMatrix(factor(pred_test), factor(true_test))
    accuracy <- conf_matrix$overall["Accuracy"]
    kappa <- conf_matrix$overall["Kappa"]
    nir <- max(conf_matrix$table) / sum(conf_matrix$table)
    p_value <- conf_matrix$overall["AccuracyPValue"]
    
    # Store results
    grid_search_results <- rbind(
      grid_search_results,
      data.frame(
        Hidden_Layers = toString(hidden_layers),
        Accuracy = accuracy,
        Kappa = kappa,
        NIR = nir,
        P_Value = p_value
      )
    )
    
    # Print metrics for the current model
    cat("Hidden Layers:", toString(hidden_layers), "\n")
    cat("Accuracy:", accuracy, "\n")
    cat("Kappa:", kappa, "\n")
    cat("No Information Rate (NIR):", nir, "\n")
    cat("P-Value [Acc > NIR]:", p_value, "\n\n")
  }
  
  # Best model based on accuracy
  best_model <- grid_search_results[which.max(grid_search_results$Accuracy), ]
  print(best_model)
  
  # Visualizing Metrics
  metrics_plot <- ggplot(grid_search_results, aes(x = Hidden_Layers, y = Accuracy, fill = Hidden_Layers)) +
    geom_bar(stat = "identity", width = 0.6) +
    labs(title = "Accuracy by Hidden Layer Configuration", y = "Accuracy", x = "Hidden Layers") +
    theme_minimal() +
    scale_fill_manual(values = rep(c("#4FC3F7", "#29B6F6", "#03A9F4", "#0288D1", "#0277BD", "#01579B"), length.out = nrow(grid_search_results))) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  print(metrics_plot)
  
  # Neural Network Plot for the Best Model
  hidden_layer_config <- as.numeric(unlist(strsplit(best_model$Hidden_Layers, ",")))
  
  # Neural Network Plot for the Best Model
  nn_best_model <- neuralnet(
    formula,
    data = train_data,
    hidden = hidden_layer_config,
    threshold = threshold,
    linear.output = FALSE,
    stepmax = 1e6,
    lifesign = "minimal"
  )
  
  # Neural Network plot
  plot(nn_best_model)
}

# Neural net library with hyperparameter 
hidden_layers_list <- list(c(6,4), c(6,3),  c(8,6), c(8,4), c(8,5), c(10,8), c(10,5), c(10,3),  c(10,5,3))
nn_model_with_hyperparameter(vehicleDataDF_encoded, hidden_layers_list,threshold = 0.01, seed = 4232,train_index,train_data,test_data)
#---------------------------------nn_model_with_hyperparameter END------------------------------------


#---------------------------------nn_model_with_k_fold_cross_validation using neuralnet library START------------------------------------
# K-Fold Cross-Validation Function
nn_model_with_k_fold_cross_validation <- function(vehicleDataDF_encoded, hidden_layers = c(8, 4), threshold = 0.01, k_values = c(5, 10), seed = 42,train_index,train_data,test_data ) {
  
  # Normalize and one-hot encode the data
  vehicleDataDF_encoded <- min_max_normalization(vehicleDataDF)
  
  vehicleDataDF_encoded <- fastDummies::dummy_cols(
    vehicleDataDF, 
    select_columns = c("Vehicle.Model", "Charging.Station.Location", "Time.of.Day", "Day.of.Week", "Charger.Type", "User.Type"), 
    remove_first_dummy = FALSE, 
    remove_selected_columns = TRUE
  )
  
  # Replace spaces and dashes in column names
  colnames(vehicleDataDF_encoded) <- gsub(" ", "_", colnames(vehicleDataDF_encoded))
  colnames(vehicleDataDF_encoded) <- gsub("-", "_", colnames(vehicleDataDF_encoded))
  
  # Normalize the encoded data again
  vehicleDataDF_encoded <- min_max_normalization(vehicleDataDF_encoded)
  
  # Check for NA values
  if (any(is.na(vehicleDataDF_encoded))) {
    stop("There are NA values in the dataset.")
  }
  
  # Split data into train and test sets
  set.seed(seed)
  #samp <- floor(0.8 * nrow(vehicleDataDF_encoded))
  #train_index <- sample(seq_len(nrow(vehicleDataDF_encoded)), size = samp)
  #train_data <- vehicleDataDF_encoded[train_index, ]
  #test_data <- vehicleDataDF_encoded[-train_index, ]
  
  # Create formula for model
  input_features <- setdiff(colnames(vehicleDataDF_encoded), grep("User.Type_", colnames(vehicleDataDF_encoded), value = TRUE))
  output_features <- grep("User.Type_", colnames(vehicleDataDF_encoded), value = TRUE)
  formula <- as.formula(paste(paste(output_features, collapse = "+"), "~", paste(input_features, collapse = "+")))
  
  # K-fold Cross-validation
  cv_accuracy <- numeric(length(k_values))
  
  for (j in 1:length(k_values)) {
    k <- k_values[j]
    folds <- cut(seq(1, nrow(train_data)), breaks = k, labels = FALSE)
    fold_accuracy <- numeric(k)
    
    for(i in 1:k) {
      # Split the data for cross-validation
      test_fold <- train_data[folds == i, ]
      train_fold <- train_data[folds != i, ]
      
      # Train Neural Network
      nn_cv <- neuralnet(
        formula, 
        data = train_fold, 
        hidden = hidden_layers, 
        linear.output = FALSE,
        stepmax = 1e6,              
        threshold = threshold,           
        lifesign = "minimal"
      )
      
      # Predictions and accuracy for cross-validation
      pr.nn_cv <- compute(nn_cv, test_fold[, input_features])  
      pred_cv <- apply(pr.nn_cv$net.result, 1, which.max)  
      true_cv <- apply(test_fold[, output_features], 1, which.max)  
      fold_accuracy[i] <- mean(pred_cv == true_cv)  
    }
    
    mean_cv_accuracy <- mean(fold_accuracy)
    cv_accuracy[j] <- mean_cv_accuracy
  }
  
  # Training accuracy
  pr.nn_train <- compute(nn_cv, train_data[, input_features])  
  pred_train <- apply(pr.nn_train$net.result, 1, which.max)  
  true_train <- apply(train_data[, output_features], 1, which.max)  
  train_accuracy <- mean(pred_train == true_train)  
  cat("Training Accuracy:", train_accuracy, "\n")
  
  # Test accuracy
  pr.nn_test <- compute(nn_cv, test_data[, input_features])  
  pred_test <- apply(pr.nn_test$net.result, 1, which.max)  
  true_test <- apply(test_data[, output_features], 1, which.max)  
  test_accuracy <- mean(pred_test == true_test)  
  cat("Test Accuracy:", test_accuracy, "\n")
  
  # Visualizing accuracies
  accuracy_data <- data.frame(
    k_values = k_values,
    Accuracy = cv_accuracy
  )
  
  ggplot(accuracy_data, aes(x = factor(k_values), y = Accuracy, fill = factor(k_values))) +
    geom_bar(stat = "identity", width = 0.6) +
    labs(title = "Cross-Validation Accuracy for Different k Values", y = "Accuracy", x = "k Value") +
    theme_minimal() +
    scale_fill_manual(values = c("#4FC3F7", "#0288D1")) +
    geom_text(aes(label = round(Accuracy, 2)), vjust = -0.3, size = 5)
}

# plotting function
plot_cross_validation_accuracy <- function(k_values, all_fold_accuracies) {
  colors <- c("#4FC3F7", "#0288D1", "#0277BD", "#66BB6A", "#388E3C", "#2C6B2F","#8D6E63", "#6D4C41", "#5D4037", "#3E2723")
  
  for (k in k_values) {
    fold_accuracies <- all_fold_accuracies[[paste0("k_", k)]]
    fold_data <- data.frame(Fold = 1:length(fold_accuracies), Accuracy = fold_accuracies)
    
    plot <- ggplot(fold_data, aes(x = factor(Fold), y = Accuracy, fill = factor(Fold))) +
      geom_bar(stat = "identity", width = 0.6) +
      labs(title = paste("Cross-Validation Accuracy for k =", k), y = "Accuracy", x = "Fold") +
      theme_minimal() +
      scale_fill_manual(values = colors[1:length(fold_accuracies)]) +  # Dynamic colors
      geom_text(aes(label = round(Accuracy, 2)), vjust = -0.3, size = 5)
    
    print(plot) 
  }
}

# Run the K-Fold Cross-Validation
result <- nn_model_with_k_fold_cross_validation(vehicleDataDF_encoded, hidden_layers = c(8, 4),threshold = 0.01, k_values = c(5, 10),seed = 42,train_index,train_data,test_data)

# Call the plotting function
plot_cross_validation_accuracy(k_values = c(5, 10), all_fold_accuracies = result$all_fold_accuracies)

#---------------------------------nn_model_with_k_fold_cross_validation using neuralnet library END------------------------------------

#---------------------------------nn_model_with nnet library START------------------------------------

nn_model_with_nnet <- function(data, target_columns, num_folds = 10, tune_length = 5) {
  # Recreating the dependent variable after one-hot encoding
  y <- factor(
    apply(data[, target_columns], 1, 
          function(row) which(row == 1))
  )
  
  # Determining independent variables
  X <- data[, !colnames(data) %in% target_columns]
  
  # Check if the dimensions are compatible
  cat("X dimensions:", dim(X), "\n")
  cat("y length:", length(y), "\n")
  
  # Cross validation settings
  train_control <- trainControl(
    method = "cv",          # Cross-validation
    number = num_folds,     # Number of folds
    verboseIter = TRUE    
  )
  
  # Training the neural network model
  nnet_model <- train(
    X, y,
    method = "nnet",        
    trControl = train_control,  
    linout = FALSE,         # Indicates that the output is categorical
    trace = FALSE,          
    tuneLength = tune_length # Number of combinations to try for hyperparameter tuning
  )
  
  # 5. Returning model results
  list(
    model = nnet_model,
    bestTune = nnet_model$bestTune,
    resample = nnet_model$resample
  )
}
# Sample data frame and target columns
data <- vehicleDataDF_encoded  
target_columns <- c("User.Type_Casual_Driver", "User.Type_Commuter", "User.Type_Long_Distance_Traveler")

# Call the function
model_results <- nn_model_with_nnet(data, target_columns)

# Print the results

print(model_results$model) # Full model results
print(model_results$bestTune) # Best hyperparameter combination
print(model_results$resample) # Cross validation results

#---------------------------------nn_model_with nnet library END------------------------------------

#//////////////////////////////////////////////////////////////////////////////////////////
#---------------------RUN THIS SECTION FOR KNN AND LOGISTIC REGRESSION------------------------------------------------------
# Select features (exclude the target column 'User.Type') for training and testing sets
features_train <- train_data[, !names(train_data) %in% "User.Type"]
features_test <- test_data[, !names(test_data) %in% "User.Type"]

# Convert categorical features into dummy variables
dummies <- dummyVars("~ .", data = features_train)
features_train_transformed <- predict(dummies, newdata = features_train)
features_test_transformed <- predict(dummies, newdata = features_test)

# Normalize the features using preProcess function
pre_proc <- preProcess(features_train_transformed, method = c("center", "scale"))
features_train_normalized <- predict(pre_proc, features_train_transformed)
features_test_normalized <- predict(pre_proc, features_test_transformed)

# Extract target labels
labels_train <- train_data$User.Type
labels_test <- test_data$User.Type

#//////////////////////////////////////////////////////////////////////////////////////////
#---------------------------------KNN------------------------------------------------------
# Cross-validation setup
set.seed(424242)
train_control_knn <- trainControl(method = "repeatedcv", number = 10, repeats = 10, search = "grid")

# Train kNN model with cross-validation and hyperparameter tuning
knn_model <- train(
  x = features_train_transformed, 
  y = labels_train, 
  method = "knn", 
  trControl = train_control_knn,
  tuneLength = 20  # Tuning over a range of k values from 1 to 20
)

# Print the results of the kNN model
print(knn_model)

# Optimal k and the best accuracy
best_k <- knn_model$bestTune$k
best_accuracy <- max(knn_model$results$Accuracy)
cat("Optimal k:", best_k, "\n")
cat("Best Accuracy:", best_accuracy, "\n")

# Plot k vs Accuracy
ggplot(knn_model$results, aes(x = k, y = Accuracy)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(title = "k vs Accuracy for k-NN Model",
       x = "k (Number of Neighbors)",
       y = "Accuracy") +
  theme_minimal()

# Final Model Performance on Test Data
# Predicting with the best k on test data
final_predictions <- predict(knn_model, newdata = features_test_transformed)

# Confusion matrix for the test set
conf_matrix <- confusionMatrix(final_predictions, labels_test)
print(conf_matrix)

# Confusion Matrix Plot
conf_matrix_data <- as.table(conf_matrix)

# Convert confusion matrix data to a dataframe
conf_matrix_df <- as.data.frame(as.table(conf_matrix_data))
colnames(conf_matrix_df) <- c("Predicted", "Actual", "Frequency")

# Plot the confusion matrix using ggplot2
ggplot(conf_matrix_df, aes(x = Actual, y = Predicted, fill = Frequency)) +
  geom_tile() +
  geom_text(aes(label = Frequency), color = "white", size = 5) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix",
       x = "Actual Labels",
       y = "Predicted Labels") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
#-------------------------------------------------------------------------------------------------------
#/////////////////////////////////////////////////////////////////////////////////////////////////////////
#---------------------------------------SVM----------------------------------------------------------
library(pROC)
shuffled_df_SVM <- vehicleDataDF_preprocessedSVM[sample(nrow(vehicleDataDF_preprocessedSVM)), ]

# Split the data into training (80%) and testing (20%) sets
train_index_SVM <- sample(1:nrow(shuffled_df_SVM), size = 0.8 * nrow(shuffled_df_SVM))
train_data_SVM <- shuffled_df[train_index_SVM, ]
test_data_SVM <- shuffled_df[-train_index_SVM, ]

# Generate a grid of 20 sigma values (e.g., from 0.01 to 2)
sigma_values <- seq(0.01, 2, length.out = 5)

# Specify the grid for C and sigma
tuneGrid <- expand.grid(
  C = c(0.1, 1, 5),  # Example values for C
  sigma = sigma_values
)

# Train the SVM model with cross-validation (10-fold)
svm_cv_model <- train(
  `User.Type` ~ ., data = train_data,
  method = "svmRadial",  # SVM with Radial basis kernel
  trControl = trainControl(method = "cv", number = 10),  # 10-fold cross-validation
  tuneGrid = tuneGrid,  # Hyperparameters for SVM (using sigma instead of gamma)
  metric = "Accuracy"  # Optimize for accuracy
)

# Print the cross-validation results
print(svm_cv_model)

plot(svm_cv_model)

# Best model (optimal hyperparameters) found during cross-validation
best_model <- svm_cv_model$finalModel
cat("Best Model Summary:\n")
summary(best_model)
# Make predictions on the test set using the cross-validated model
svm_cv_predictions <- predict(svm_cv_model, newdata = test_data)

# Evaluate the model's performance using confusion matrix
svm_cv_confusion_matrix <- confusionMatrix(svm_cv_predictions, test_data$`User.Type`)

# Print confusion matrix and accuracy
print(svm_cv_confusion_matrix)
cat("Accuracy (Cross-validated model):", svm_cv_confusion_matrix$overall["Accuracy"], "\n")

# Extract confusion matrix values for each class
conf_matrix <- svm_cv_confusion_matrix

# For Class: Casual Driver
TP_Casual <- conf_matrix$table["Casual Driver", "Casual Driver"]
FP_Casual <- sum(conf_matrix$table["Casual Driver", ]) - TP_Casual
FN_Casual <- sum(conf_matrix$table[, "Casual Driver"]) - TP_Casual
TN_Casual <- sum(conf_matrix$table) - (TP_Casual + FP_Casual + FN_Casual)

# Calculate Precision, Recall, F1 Score, Specificity for Casual Driver
precision_Casual <- TP_Casual / (TP_Casual + FP_Casual)
recall_Casual <- TP_Casual / (TP_Casual + FN_Casual)
f1_score_Casual <- 2 * (precision_Casual * recall_Casual) / (precision_Casual + recall_Casual)
specificity_Casual <- TN_Casual / (TN_Casual + FP_Casual)

cat("Casual Driver - Precision:", precision_Casual, "\n")
cat("Casual Driver - Recall:", recall_Casual, "\n")
cat("Casual Driver - F1 Score:", f1_score_Casual, "\n")
cat("Casual Driver - Specificity:", specificity_Casual, "\n\n")

# For Class: Commuter
TP_Commuter <- conf_matrix$table["Commuter", "Commuter"]
FP_Commuter <- sum(conf_matrix$table["Commuter", ]) - TP_Commuter
FN_Commuter <- sum(conf_matrix$table[, "Commuter"]) - TP_Commuter
TN_Commuter <- sum(conf_matrix$table) - (TP_Commuter + FP_Commuter + FN_Commuter)

# Calculate Precision, Recall, F1 Score, Specificity for Commuter
precision_Commuter <- TP_Commuter / (TP_Commuter + FP_Commuter)
recall_Commuter <- TP_Commuter / (TP_Commuter + FN_Commuter)
f1_score_Commuter <- 2 * (precision_Commuter * recall_Commuter) / (precision_Commuter + recall_Commuter)
specificity_Commuter <- TN_Commuter / (TN_Commuter + FP_Commuter)

cat("Commuter - Precision:", precision_Commuter, "\n")
cat("Commuter - Recall:", recall_Commuter, "\n")
cat("Commuter - F1 Score:", f1_score_Commuter, "\n")
cat("Commuter - Specificity:", specificity_Commuter, "\n\n")

# For Class: Long-Distance Traveler
TP_LDT <- conf_matrix$table["Long-Distance Traveler", "Long-Distance Traveler"]
FP_LDT <- sum(conf_matrix$table["Long-Distance Traveler", ]) - TP_LDT
FN_LDT <- sum(conf_matrix$table[, "Long-Distance Traveler"]) - TP_LDT
TN_LDT <- sum(conf_matrix$table) - (TP_LDT + FP_LDT + FN_LDT)

# Calculate Precision, Recall, F1 Score, Specificity for Long-Distance Traveler
precision_LDT <- TP_LDT / (TP_LDT + FP_LDT)
recall_LDT <- TP_LDT / (TP_LDT + FN_LDT)
f1_score_LDT <- 2 * (precision_LDT * recall_LDT) / (precision_LDT + recall_LDT)
specificity_LDT <- TN_LDT / (TN_LDT + FP_LDT)

cat("Long-Distance Traveler - Precision:", precision_LDT, "\n")
cat("Long-Distance Traveler - Recall:", recall_LDT, "\n")
cat("Long-Distance Traveler - F1 Score:", f1_score_LDT, "\n")
cat("Long-Distance Traveler - Specificity:", specificity_LDT, "\n")

# Calculate AUC using ROC curve
roc_curve <- roc(test_data$`User.Type`, as.numeric(svm_cv_predictions))
auc_value <- auc(roc_curve)
cat("AUC:", auc_value, "\n")

# Confusion Matrix Plot (Optional, if needed)
conf_matrix_data_svm <- as.table(svm_cv_confusion_matrix)
conf_matrix_df_svm <- as.data.frame(as.table(conf_matrix_data_svm))
colnames(conf_matrix_df_svm) <- c("Predicted", "Actual", "Frequency")

# Plot confusion matrix using ggplot2
ggplot(conf_matrix_df_svm, aes(x = Actual, y = Predicted, fill = Frequency)) +
  geom_tile() +
  geom_text(aes(label = Frequency), color = "white", size = 5) +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Confusion Matrix", x = "Actual Labels", y = "Predicted Labels") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ----------------------------------- 3. Accuracy Plot for Different Hyperparameters --------------------------------------

# Extract accuracy for each fold
accuracy_per_fold <- svm_cv_model$resample$Accuracy
fold_numbers <- seq_along(accuracy_per_fold)

# Create a data frame for plotting
accuracy_vs_fold <- data.frame(
  Fold = fold_numbers,
  Accuracy = accuracy_per_fold
)

# Plot accuracy vs fold
library(ggplot2)

ggplot(accuracy_vs_fold, aes(x = Fold, y = Accuracy)) +
  geom_line(color = "orange", size = 1) +
  geom_point(color = "black", size = 3) +
  theme_minimal() +
  labs(
    title = "Accuracy vs Fold",
    x = "Fold Number",
    y = "Accuracy"
  ) +
  theme(
    text = element_text(size = 12),
    axis.title = element_text(face = "bold"),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

#--------------------------------------------------------------
# Extract the results from the trained SVM model
results <- svm_cv_model$results

# Ensure that the C values are factors for facetting
results$C <- factor(results$C)

# Find the maximum accuracy for each C
max_accuracy_data <- results %>%
  group_by(C) %>%
  summarise(max_accuracy = max(Accuracy)) %>%
  ungroup()

# Plot Accuracy vs Sigma for Different C values
ggplot(results, aes(x = sigma, y = Accuracy, color = C)) +
  geom_line() +
  geom_point() +
  facet_wrap(~ C, scales = "free", ncol = 3) +  # Create separate plots for each C value
  labs(title = "Accuracy vs Sigma for Different C Values",
       x = "Sigma",
       y = "Accuracy") +
  theme_minimal() +
  theme(legend.position = "none") +  # Remove legend if not necessary
  geom_text(data = max_accuracy_data, 
            aes(x = 0.01, y = max_accuracy+0.01, label = round(max_accuracy, 3)), 
            color = "black",  vjust = -0.1)
#---------------------------------------------------------------------------------------------------
#//////////////////////////////////////////////////////////////////////////////////////////////////////
#-------------------------------LOGISTIC REGRESSION MODEL------------------------------------------------
# Load the necessary library for multinomial logistic regression
library(nnet)
# Define cross-validation parameters (e.g., 10-fold cross-validation)
train_control_lr <- trainControl(method = "cv", number = 10)
cv_model <- train(User.Type ~ ., 
                  data = data.frame(features_train_normalized, User.Type = labels_train),
                  method = "multinom", 
                  trControl = train_control_lr)

# Print the cross-validation results
print(cv_model)

# Make predictions on the test set (after cross-validation)
cv_predictions <- predict(cv_model, newdata = data.frame(features_test_normalized))

# Confusion matrix for the test set
conf_matrix_cv <- confusionMatrix(factor(cv_predictions, levels = levels(labels_test)), labels_test)

# Print confusion matrix
print(conf_matrix_cv)
#-------------------------------plotting----------------------------------------
# Print accuracy
cat("Accuracy (Multinomial Logistic Regression with Cross-Validation):", conf_matrix_cv$overall["Accuracy"], "\n")

# Create dot plot for accuracy across cross-validation folds
ggplot(cv_model$resample, aes(x = Resample, y = Accuracy)) +
  geom_point(color = "black", size = 3) + 
  geom_jitter(width = 0.01, height = 0) +  # Add jitter for better visualization
  labs(title = "Cross-validation Accuracy for Multinomial Logistic Regression",
       x = "Cross-validation Fold", y = "Accuracy") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Make predictions on the test set
cv_predictions <- predict(cv_model, newdata = data.frame(features_test_normalized))

# Confusion matrix for the test set
conf_matrix_cv <- confusionMatrix(factor(cv_predictions, levels = levels(labels_test)), labels_test)

# Print confusion matrix
print(conf_matrix_cv)

# Print accuracy
cat("Accuracy (Multinomial Logistic Regression with Cross-Validation):", conf_matrix_cv$overall["Accuracy"], "\n")


# Make predictions on the test set (after cross-validation)
cv_predictions <- predict(cv_model, newdata = data.frame(features_test_normalized))

# Create a data frame of actual vs predicted values
actual_vs_predicted <- data.frame(Actual = factor(labels_test), Predicted = factor(cv_predictions))

# Plot Actual vs Predicted using ggplot2 (Bar Plot)
ggplot(actual_vs_predicted, aes(x = Actual, fill = Predicted)) +
  geom_bar(position = "fill", stat = "count") +
  labs(title = "Actual vs Predicted Values for Multinomial Logistic Regression",
       x = "Actual User Type", y = "Proportion") +
  theme_minimal() +
  scale_fill_manual(values = c("Casual Driver" = "darkblue", "Commuter" = "orange", "Long-Distance Traveler" = "darkred")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
#//////////////////////////////////////////////////////////////////////////////////////////


# Common Cross-Validation Control
train_control <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = "final",
  classProbs = TRUE,
  summaryFunction = multiClassSummary  # For multiclass metrics
)

# -----------------------------------
# 1. Naive Bayes Model
naive_bayes_model <- train(
  User.Type ~ ., 
  data = train_data, 
  method = "naive_bayes", 
  trControl = train_control
)

# -----------------------------------
# 2. Decision Tree Model
tune_grid <- expand.grid(
  cp = seq(0.001, 0.1, by = 0.01)  #  step size for finer tuning
)
decision_tree_model <- train(
  User.Type ~ ., 
  data = train_data, 
  method = "rpart", 
  trControl = train_control, 
  tuneGrid = tune_grid
)
# -----------------------------------
# 3. Random Forest Model
random_forest_model <- train(
  User.Type ~ ., 
  data = train_data, 
  method = "rf", 
  trControl = train_control
)

# -----------------------------------
# # Cross-Validation Results
# cat("Cross-Validation Results:\n")
# results_list <- list(
#   Naive_Bayes = naive_bayes_model$results,
#   Decision_Tree = decision_tree_model$results,
#   Random_Forest = random_forest_model$results
# )
# print(results_list)

# -----------------------------------
# Function to evaluate model and return both train and test metrics
evaluate_model <- function(model, train_data, test_data, target, model_name) {
  # Train predictions
  train_predictions <- predict(model, newdata = train_data)
  train_prob_predictions <- predict(model, newdata = train_data, type = "prob")
  
  # Test predictions
  test_predictions <- predict(model, newdata = test_data)
  test_prob_predictions <- predict(model, newdata = test_data, type = "prob")
  
  
  # Confusion Matrix for Train and Test Data
  train_conf_matrix <- confusionMatrix(train_predictions, train_data[[target]])
  test_conf_matrix <- confusionMatrix(test_predictions, test_data[[target]])
  
  
  
  # Train Metrics
  train_by_class <- train_conf_matrix$byClass
  train_accuracy <- train_conf_matrix$overall["Accuracy"]
  train_precision <- mean(train_by_class[,"Precision"], na.rm = TRUE)
  train_recall <- mean(train_by_class[,"Recall"], na.rm = TRUE)
  train_f1 <- mean(train_by_class[,"F1"], na.rm = TRUE)
  train_roc_auc <- auc(multiclass.roc(train_data[[target]], train_prob_predictions))
  
  # Test Metrics
  test_by_class <- test_conf_matrix$byClass
  test_accuracy <- test_conf_matrix$overall["Accuracy"]
  test_precision <- mean(test_by_class[,"Precision"], na.rm = TRUE)
  test_recall <- mean(test_by_class[,"Recall"], na.rm = TRUE)
  test_f1 <- mean(test_by_class[,"F1"], na.rm = TRUE)
  test_roc_auc <- auc(multiclass.roc(test_data[[target]], test_prob_predictions))
  
  # Combine Train and Test Metrics into one table
  metrics_df <- data.frame(
    Metric = c("Accuracy", "Precision", "Recall", "F1", "ROC AUC"),
    Train = c(train_accuracy, train_precision, train_recall, train_f1, train_roc_auc),
    Test = c(test_accuracy, test_precision, test_recall, test_f1, test_roc_auc)
  )
  list(
    Metrics = metrics_df,
    ConfusionMatrix = test_conf_matrix$table
  )
  # Return metrics table for the model
  #return(metrics_df)
}

# Function to print the metrics table
print_model_metrics <- function(metrics_df, model_name) {
  if (is.null(metrics_df) || nrow(metrics_df) == 0) {
    stop("Metrics table is empty. Ensure evaluate_model is generating a valid table.")
  }
  
  metrics_df %>%
    kable("html", caption = paste(model_name, "- Metrics")) %>%
    kable_styling(
      bootstrap_options = c("striped", "hover", "condensed"),
      full_width = F,
      position = "center"
    ) %>%
    column_spec(1, color = "black", bold = TRUE) %>%
    column_spec(2, color = "black", width = "3cm",border_left = TRUE) %>%
    column_spec(3, color = "black", width = "3cm", border_left = TRUE) %>%
    row_spec(0, bold = TRUE, color = "black", background = "lightgray") %>%
    row_spec(seq_len(nrow(metrics_df)), color = "black", background = "white")
}
metrics_nb <- evaluate_model(naive_bayes_model, train_data, test_data, "User.Type", "Naive Bayes")
metrics_dt <- evaluate_model(decision_tree_model, train_data, test_data, "User.Type", "Decision Tree")
metrics_rf <- evaluate_model(random_forest_model, train_data, test_data, "User.Type", "Random Forest")
# Now, print the metrics tables for each model
print_model_metrics(metrics_nb$Metrics, "Naive Bayes")
print_model_metrics(metrics_dt$Metrics, "Decision Tree")
print_model_metrics(metrics_rf$Metrics, "Random Forest")


#-----------------printing confusion matrices-----------------

conf_matrix_nb <- as.data.frame(metrics_nb$ConfusionMatrix)
conf_matrix_dt <- as.data.frame(metrics_dt$ConfusionMatrix)
conf_matrix_rf <- as.data.frame(metrics_rf$ConfusionMatrix)

# Function to plot the confusion matrix
plot_confusion_matrix <- function(conf_matrix_table, title) {
  ggplot(conf_matrix_table, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "black", size = 5) +
    scale_fill_gradient(low = "violet", high = "purple") +
    theme_minimal() +
    labs(title = title, x = "Predicted", y = "Actual")
}

# Plot confusion matrices for all models
plot_confusion_matrix(conf_matrix_nb, "Naive Bayes Confusion Matrix")
plot_confusion_matrix(conf_matrix_dt, "Decision Tree Confusion Matrix")
plot_confusion_matrix(conf_matrix_rf, "Random Forest Confusion Matrix") 
















