library(shiny)
library(leaflet)
library(RColorBrewer)
library(scales)
library(lattice)
library(dplyr)
library(DT)
library(rCharts)
library(rjson)


path <- getwd()
makepath  <- function(filename){
  paste0(getwd(), '/', filename )
}

restaurants=data.frame()
files=c("data/operating_hours_Mon.csv","data/operating_hours_Tue.csv",
        "data/operating_hours_Wed.csv","data/operating_hours_Thu.csv",
        "data/operating_hours_Fri.csv","data/operating_hours_Sat.csv",
        "data/operating_hours_Sun.csv")
days=c('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday')

getHour <- function(x){
  
  if (x=='0:00 am'){return(0)}
  if (x=='0:00 pm'){return(12)}
  
  my_time= strptime(x, "%I:%M %p")
  my_hour=as.numeric(strftime(my_time, format="%H"))+as.numeric(strftime(my_time, format="%M"))/60
  my_hour
}

for(i in seq(1:7)) {
  #rbind other days as well
  restaurants_dummy=read.csv(files[i])
  restaurants_dummy$opening_1_hours=sapply(restaurants_dummy[,c('opening_1')], function(x) getHour(x) )
  restaurants_dummy$closing_1_hours=sapply(restaurants_dummy[,c('closing_1')], function(x) getHour(x) )
  restaurants_dummy$day=days[i]
  restaurants=rbind(restaurants,restaurants_dummy)
}
restaurants=subset(restaurants, ! (is.na( closing_1_hours) | is.na(opening_1_hours)))
restaurants=subset(restaurants, ! (is.na( latitude) | is.na(longitude)))

crime.data <- read.csv("crime-data.csv")
crime.data <- subset(crime.data, select = c("Weekday", "Crime.Type", "Latitude", "Longitude", "Date..Time"))
crime.data$Date..Time <- as.POSIXlt(crime.data$Date..Time)$hour

