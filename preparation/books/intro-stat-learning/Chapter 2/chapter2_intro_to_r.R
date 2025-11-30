# Create vectors
x <- c(1, 3, 2, 5)
y <- c(1, 4, 3)

# Check lengths
length(x)
length(y)

# Add vectors (element-wise)
x + y

# List objects in workspace
ls()

# Remove specific objects
rm(x, y)

# Remove all objects
rm(list = ls())



### Matrices

# Create a matrix (column-wise default)
x <- matrix(c(1, 2, 3, 4), nrow = 2, ncol = 2)

# Create a matrix (row-wise)
matrix(c(1, 2, 3, 4), 2, 2, byrow = TRUE)

# Element-wise operations
sqrt(x)
x^2


### Random Numbers & Statistics

# Reproducible random numbers
set.seed(1303)
rnorm(50)

# Correlation example
set.seed(1)
x <- rnorm(50)
y <- x + rnorm(50, mean = 50, sd = 0.1)
cor(x, y)

# Basic statistics
set.seed(3)
y <- rnorm(100)
mean(y)
var(y)
sd(y)          # same as sqrt(var(y))



### Graphics

set.seed(10)
x <- rnorm(100)
y <- rnorm(100)

# Basic scatterplot
plot(x, y)

# Scatterplot with labels and title
plot(
  x, y,
  xlab = "This is the x-axis",
  ylab = "This is the y-axis",
  main = "Plot of X vs Y"
)

# Save plot as PDF
pdf("Figure.pdf")
plot(x, y, col = "green")
dev.off()

# Sequences
seq(1, 10)
1:10
seq(-pi, pi, length = 50)

# Advanced: contour, image, persp plots
x <- seq(-pi, pi, length = 50)
y <- x
f <- outer(x, y, function(x, y) cos(y) / (1 + x^2))

contour(x, y, f)
contour(x, y, f, nlevels = 45, add = TRUE)

fa <- (f - t(f)) / 2

image(x, y, fa)
persp(x, y, fa)
persp(x, y, fa, theta = 30, phi = 20)



### Indexing

A <- matrix(1:16, 4, 4)

A[2, 3]           # single element
A[c(1, 3), c(2, 4)]  # select rows/columns
A[1:3, 2:4]          # ranges
A[1:2, ]             # all columns for rows 1–2
A[, 1:2]             # all rows, columns 1–2
A[-c(1, 3), ]        # exclude rows
A[-c(1, 3), -c(1, 3, 4)]  # exclude rows/columns

dim(A)



### Loading Data

# Load tabular data
Auto <- read.table("Auto.data", header = TRUE, na.strings = "?")

# Remove missing rows
Auto <- na.omit(Auto)

# View dataset
names(Auto)
dim(Auto)

# CSV loading
Auto <- read.csv("Auto.csv", header = TRUE, na.strings = "?")


### Additional Plots & Summaries

# Using $ to reference columns
plot(Auto$cylinders, Auto$mpg)

# Make variables accessible without $
attach(Auto)

# Convert numeric → categorical
cylinders <- as.factor(cylinders)

# Boxplots for categorical variables
plot(cylinders, mpg, col = "red", varwidth = TRUE,
     xlab = "Cylinders", ylab = "MPG")

# Histogram
hist(mpg, col = 2, breaks = 15)

# Scatterplot matrix
pairs(Auto)
pairs(~ mpg + displacement + horsepower + weight + acceleration, Auto)

# Identify points on a plot
plot(horsepower, mpg)
identify(horsepower, mpg, name)

# Summary statistics
summary(Auto)
summary(mpg)



### Exiting and Saving

savehistory("session_history.Rhistory")   # save command history
# loadhistory("session_history.Rhistory") # load it later
# q()  # quit R
