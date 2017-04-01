var margin = {top: 20, right: 20, bottom: 30, left: 40},
    width = 960 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;
console.log("It comes here")    

var svg = d3.select("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

var tooltip = d3.select("body").append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);
var xScale = d3.scale.linear()
              .domain([d3.min(xVal), 0.1])
              .range([ 0, width ]);
    
var yScale = d3.scale.linear()
    	      .domain([d3.min(yVal), d3.max(yVal)])
    	      .range([ height, 0 ]);
var xAxis = d3.svg.axis()
			.scale(xScale)
			.orient("bottom")
			.ticks(21);
 color = d3.scale.category10();			

var	yAxis = d3.svg.axis()
			.scale(yScale)
			.orient("left")
			.ticks(21);

svg.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis)
    .append("text")
      .attr("class", "label")
      .attr("x", width)
      .attr("y", -6)
      .style("text-anchor", "end")
      .text("Principal Component1");  

  // y-axis
svg.append("g")
      .attr("class", "y axis")
      .call(yAxis)
    .append("text")
      .attr("class", "label")
      .attr("transform", "rotate(-90)")
      .attr("y", 6)
      .attr("dy", ".71em")
      .style("text-anchor", "end")
      .text("Principal COmponent 2");        	      

svg.selectAll(".dot")
      .data(d3.zip(xVal,yVal))
    .enter().append("circle")
      .attr("class", "dot")
      .transition()
      .duration(1000)
      .each("start", function() { 
                d3.select(this)
                .attr("fill", "red") 
                .attr("r", 5);
      })
      .delay(function(d, i) {
            return i / xVal.length * 500;
        })
      .attr("r", 3.5)
      .attr("cx", function(d,i) { return xScale(d[0]);})
      .attr("cy", function(d) { return yScale(d[1]);})
      .each("end", function() {
            d3.select(this) 
             .transition()
             .duration(500)
             .attr("fill", "black")  
             .attr("r", 2); 
       });
svg.selectAll(".dot")
	 .on("mouseover", function(d,i) {
      		console.log(d[0]);
          tooltip.transition()
               .duration(200)
               .style("opacity", .9);
          tooltip.html( "<br/> (" + d[0] 
	        + ", " + d[1] + ")")
               .style("left", (d3.event.pageX + 5) + "px")
               .style("top", (d3.event.pageY - 28) + "px");
      })
      .on("mouseout", function(d) {
          tooltip.transition()
               .duration(500)
               .style("opacity", 0);
      });