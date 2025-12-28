smart_source <- function(file) {
  lines <- readLines(file)
  
  # Echo only non-print statements
  for (i in seq_along(lines)) {
    line <- lines[i]
    if (!grepl("^\\s*(print|cat)\\s*\\(", line)) {
      cat(line, "\n")
    }
  }
  
  # Run normally
  source(file, echo = FALSE, print.eval = TRUE)
}
