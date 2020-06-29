
# To produce html output automatically.


weekdays <- c("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
output_file <- paste0(weekdays, ".html")
params <- lapply(weekdays, FUN = function(x){list(weekday=x)})

reports <- tibble(output_file, params)

# tibble is really flexible, params column actually each is a list.
#reports <- tibble(output_file, params=teamID)



library(rmarkdown)
# need to use x[[1]] to get at elememets since tibble doesn't simplify
apply(reports, MARGIN=1, 
      FUN=function(x){  # x[[1]] since tibble does not simplify.
        render(input = "ST558project2.Rmd", output_file = x[[1]], params = x[[2]])
      })