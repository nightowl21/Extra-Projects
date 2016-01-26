source("helper_file.R")


icons <- iconList(
  active = makeIcon(makepath("active.png"), iconWidth=18, iconHeight=18),
  arts = makeIcon(makepath("art-and-culture.png"),  iconWidth=24, iconHeight=24),
  restaurants = makeIcon(makepath("dinner-eat-restaurant-icon.png"), 
                         iconWidth=24, iconHeight=24),
  nightlife = makeIcon(makepath("glass_icon1.png"), iconWidth=24, iconHeight=24)
)

shinyServer(function(input, output, session) {

  ## Interactive Map ###########################################

  # Create the map
  output$map <- renderLeaflet({
    leaflet() %>%
      addTiles(
        urlTemplate = "//{s}.tiles.mapbox.com/v3/jcheng.map-5ebohr46/{z}/{x}/{y}.png",
        attribution = 'Maps by <a href="http://www.mapbox.com/">Mapbox</a>'
      ) %>%
      setView(-122.42, 37.78, zoom = 13) 
  })


  # This observer is responsible for maintaining the icon markers and legend,
  # according to the variables the user has chosen to map to color and size.
  observeEvent(input$go, { 
    data1 = restaurants[restaurants$day == input$var, ]
    data <- subset(data1, category %in% input$choices)
    
    myday <- input$var
    mytime <- input$bins
    mychoices <- input$choices

    data <- subset(data, (data$opening_1_hours<= mytime) & 
                          (data$closing_1_hours> mytime))
    if (nrow(data)==0){
      leafletProxy("map", data = data) %>%
        clearMarers()
    } else {
    leafletProxy("map", data = data) %>% 
      clearMarkers() %>%
      addMarkers(icon = ~icons[category], 
                 popup = ~paste0("<b><h4>", name, "</b></h4>",
                                 "<b>Category:</b> ", category, "<br>",
                                 "<b>Timings:</b> ", opening_1, ' - ', closing_1,
                                 "<br>", 
                                 "<b>Price:</b> ", price_range, "<br>",
                                 "<b>Ratings:</b> ", rating, 
                                 " (", reviews, " reviews)", "<br>",
                                 "<a href=", url, ">Yelp Link</a>")
                 )
      }
    
})
  
  
  observeEvent(input$go, {
  data2 <- subset(crime.data, subset = ((Date..Time == floor(input$bins)) & 
                                          Weekday == input$var))
  crime_array <- toJSONArray2(subset(data2, select = c("Latitude", "Longitude")), 
                              json = FALSE, names = FALSE) 
  
  leafletProxy("map", data = data2) %>% 
    clearMarkerClusters() %>%
    addCircleMarkers(popup = data2$Crime.Type, 
                     clusterOptions = markerClusterOptions(lng = data2$Latitude, 
                                                     lat = data2$Longitude))
  })
  

}) 
  