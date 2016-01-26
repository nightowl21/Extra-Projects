library(shiny)
library(leaflet)

# Choices for drop-downs
vars <- c(
  "Monday" = "Monday",
  "Tuesday" = "Tuesday",
  "Wednesday" ="Wednesday",
  "Thursday" = "Thursday",
  "Friday" = "Friday",
  "Saturday" = "Saturday" ,
  "Sunday" = "Sunday"
)


shinyUI(navbarPage("SF 360", id="nav",

  tabPanel("Interactive map",
    div(class="outer",

      tags$head(
        # Include our custom CSS
        includeCSS("styles.css"),
        includeScript("gomap.js")
      ),

        leafletOutput("map", width="100%", height="100%"),

      # Shiny versions prior to 0.11 should use class="modal" instead.
        absolutePanel(id = "controls", class = "panel panel-default",
                      fixed = TRUE, draggable = TRUE, top = 60, left = "auto", 
                      right = 20, bottom = "auto",
        width = 330, height = "auto",

        h2("Things to do"),

        selectInput("var", 
                    label = "Choose A Day",
                    choices = c('Monday','Tuesday','Wednesday','Thursday',
                                'Friday','Saturday','Sunday'),
                    selected = "Wednesday"),
        sliderInput("bins", "Time of the day (0-23):", 
                    min = 0, max = 23, value = 12, step = 1,
                    format="#.#",animate=
                      animationOptions(interval=3000, loop=TRUE)),
        checkboxGroupInput("choices", "Options",
                           choices = c(
                             Restaurants = "restaurants",
                             NightLife = "nightlife",
                             ActiveLife = "active",
                             Arts = "arts"),
                           selected = c(
                             Restaurants = "restaurants",
                             NightLife = "nightlife",
                             ActiveLife = "active",
                             Arts = "arts")),
        actionButton(inputId = "go", label = "Update"),
        p(" "),
        p("Select Time, Day and the category of Outing you are interested in and then hit Update. See the locations that are open for your entertainment. The circles with number gives the crime information about the area.")
     )
    )
  ),
  
  tabPanel("About SF 360",
           fluidPage(
             mainPanel(
                  p("People of San Francisco love to go out and this app helps them do just that. The unique feature of this app is that you can plan your future outings at a future data and time. You also get information about how safe that area is at the a certain time and day of week. Plan your outing and be safe, that is what this app is all about."),
                  
                  p("The app should locations for Restaurants, Arts and Culture, Night Life and Active Sports that are open at a chosen time of a chosen day. Click on any icon and see more details about the location. The app also shows the number of crime incidents that have have been historically recorded in that area. The crime data is based on data for January 2015 - March 2015."),
                  
                  p("Data for various locations has been been taked from Yelp. Python and Beautiful Soup (Python Web Scrapping library) were used to scrape the webpage. Data about crime has been taken from https://data.sfgov.org/. The application is built using R, Shiny and Leaflet."),
                  
                  p("This app was created by Chhavi Choudhury."),
                  a(href = "mailto:chhavichoudhury@gmail.com", "chhavichoudhury@gmail.com")
                  )
             )
           )
  )
)
