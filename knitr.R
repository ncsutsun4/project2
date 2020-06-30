
# To produce md output automatically.

# Create weekday parameter
weekdays <- c("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

# Create knitr automation parameter and out file names and put into a tibble
output_file <- paste0(weekdays, ".md")
params <- lapply(weekdays, FUN = function(x){list(weekday=x)})
reports <- tibble(output_file, params)

# Knit file and create reports automatically
apply(reports, MARGIN=1, 
      FUN=function(x){  # x[[1]] since tibble does not simplify.
        rmarkdown::render(input = "ST558project2.Rmd", output_file = x[[1]], params = x[[2]])
      })