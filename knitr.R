# Create knitr automation parameter and out file names and put into a tibble
output_file <- paste0(weekdays, "Analysis.md")
params <- lapply(weekdays, FUN = function(x){list(weekday=x)})
reports <- tibble(output_file, params)

# Knit file and create reports automatically
apply(reports, MARGIN=1, 
      FUN=function(x){  # x[[1]] since tibble does not simplify.
        rmarkdown::render(input = "ST558project2.Rmd", output_file = x[[1]],
                          params = x[[2]], clean=FALSE, envir=new.env())
      })